"""Chronological xLSTM teacher for transient memristor modeling."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from data.memristor_generator import require_sequence_tensor


def _validate_time(V: torch.Tensor, t: torch.Tensor, input_dim: int) -> None:
    require_sequence_tensor(V, name="V")
    require_sequence_tensor(t, name="t", feature_size=1)
    if V.shape[:2] != t.shape[:2]:
        raise ValueError("V and t must share batch and sequence dimensions")
    if V.shape[-1] + t.shape[-1] != input_dim:
        raise ValueError(
            f"concatenated V/t feature size must be {input_dim}, "
            f"got {V.shape[-1] + t.shape[-1]}"
        )
    if bool(torch.any(t[:, 1:] <= t[:, :-1])):
        raise ValueError("timesteps must be strictly increasing within each trajectory")


class SimplifiedSLSTMCell(nn.Module):
    """One internal sLSTM timestep. Public training uses ``xLSTMTeacher.forward``."""

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 1):
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_heads < 1:
            raise ValueError("cell dimensions must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.W_gates = nn.Linear(input_size, 4 * hidden_size)
        self.R_gates = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        nn.init.zeros_(self.R_gates.weight)
        with torch.no_grad():
            self.W_gates.bias[hidden_size : 2 * hidden_size] = torch.linspace(
                3.0, 6.0, hidden_size
            )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.ndim != 2 or x.shape[-1] != self.input_size:
            raise ValueError(
                f"cell input must have shape [batch, {self.input_size}]"
            )
        batch_size = x.shape[0]
        if state is None:
            hidden = x.new_zeros(batch_size, self.hidden_size)
            memory = x.new_zeros(batch_size, self.hidden_size)
        else:
            hidden, memory = state
            expected = (batch_size, self.hidden_size)
            if hidden.shape != expected or memory.shape != expected:
                raise ValueError("invalid sLSTM state shape")
        gates = self.W_gates(x) + self.R_gates(hidden)
        input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
        input_gate = torch.exp(input_gate.clamp(max=10.0))
        forget_gate = torch.sigmoid(forget_gate)
        candidate = torch.tanh(candidate)
        output_gate = torch.sigmoid(output_gate)
        next_memory = forget_gate * memory + input_gate * candidate
        next_hidden = output_gate * torch.tanh(next_memory)
        return next_hidden, (next_hidden, next_memory)


class SimplifiedMLSTMCell(nn.Module):
    """One internal matrix-memory timestep."""

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4):
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_heads < 1:
            raise ValueError("cell dimensions must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        self.igate = nn.Linear(input_size, num_heads)
        self.fgate = nn.Linear(input_size, num_heads)
        self.out_norm = nn.LayerNorm(hidden_size)
        with torch.no_grad():
            nn.init.zeros_(self.fgate.weight)
            self.fgate.bias.copy_(torch.linspace(3.0, 6.0, num_heads))
            nn.init.zeros_(self.igate.weight)
            nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if x.ndim != 2 or x.shape[-1] != self.input_size:
            raise ValueError(
                f"cell input must have shape [batch, {self.input_size}]"
            )
        batch_size = x.shape[0]
        q = self.W_q(x).view(batch_size, self.num_heads, self.head_dim)
        k = self.W_k(x).view(batch_size, self.num_heads, self.head_dim)
        v = self.W_v(x).view(batch_size, self.num_heads, self.head_dim)
        k = k / (torch.linalg.vector_norm(k, dim=-1, keepdim=True) + 1e-8)
        input_gate = torch.exp(self.igate(x).clamp(max=10.0)).unsqueeze(-1).unsqueeze(-1)
        forget_gate = torch.sigmoid(self.fgate(x)).unsqueeze(-1).unsqueeze(-1)
        if state is None:
            matrix = x.new_zeros(
                batch_size, self.num_heads, self.head_dim, self.head_dim
            )
            normalizer = x.new_zeros(batch_size, self.num_heads, self.head_dim, 1)
            stabilizer = x.new_zeros(batch_size, self.num_heads, 1, 1)
        else:
            matrix, normalizer, stabilizer = state
            if matrix.shape != (
                batch_size,
                self.num_heads,
                self.head_dim,
                self.head_dim,
            ):
                raise ValueError("invalid mLSTM matrix-state shape")
            if normalizer.shape != (batch_size, self.num_heads, self.head_dim, 1):
                raise ValueError("invalid mLSTM normalizer-state shape")
            if stabilizer.shape != (batch_size, self.num_heads, 1, 1):
                raise ValueError("invalid mLSTM stabilizer-state shape")
        next_matrix = forget_gate * matrix + input_gate * (
            v.unsqueeze(-1) @ k.unsqueeze(-2)
        )
        next_normalizer = forget_gate * normalizer + input_gate * k.unsqueeze(-1)
        next_stabilizer = forget_gate * stabilizer + input_gate
        numerator = torch.matmul(next_matrix, q.unsqueeze(-1)).squeeze(-1)
        denominator = (
            torch.matmul(next_normalizer.transpose(-2, -1), q.unsqueeze(-1))
            + next_stabilizer
        )
        denominator = denominator.squeeze(-1).squeeze(-1).unsqueeze(-1)
        denominator = torch.where(
            denominator.abs() < 1e-6,
            torch.full_like(denominator, 1e-6),
            denominator,
        )
        hidden = self.out_norm((numerator / denominator).reshape(batch_size, self.hidden_size))
        return hidden, (next_matrix, next_normalizer, next_stabilizer)


class xLSTMTeacher(nn.Module):
    """Sequence-only xLSTM teacher with explicit recurrent state threading."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        use_mlstm: bool = True,
        num_heads: int = 4,
    ):
        super().__init__()
        if input_dim < 2 or hidden_size < 1 or num_layers < 1 or output_dim < 1:
            raise ValueError("model dimensions are invalid")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.use_mlstm = use_mlstm
        self.input_proj = nn.Linear(input_dim, hidden_size)
        cells: list[nn.Module] = []
        for layer_index in range(num_layers):
            if use_mlstm and layer_index % 2 == 1:
                cells.append(SimplifiedMLSTMCell(hidden_size, hidden_size, num_heads))
            else:
                cells.append(SimplifiedSLSTMCell(hidden_size, hidden_size, num_heads))
        self.lstm_cells = nn.ModuleList(cells)
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def _step(self, x_t: torch.Tensor, states: list | None):
        if states is None:
            states = [None] * self.num_layers
        if len(states) != self.num_layers:
            raise ValueError(f"states must contain {self.num_layers} layer states")
        hidden = self.input_proj(x_t)
        next_states = []
        for cell, state in zip(self.lstm_cells, states):
            hidden, next_state = cell(hidden, state)
            next_states.append(next_state)
        return self.output_proj(hidden), next_states, hidden

    def forward(
        self,
        V: torch.Tensor,
        t: torch.Tensor,
        states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Process a complete ordered sequence.

        ``V`` and ``t`` must be three-dimensional, with sequence length greater
        than one. State is advanced once per chronological timestep.
        """
        _validate_time(V, t, self.input_dim)
        x = torch.cat((V, t), dim=-1)
        outputs = []
        next_states = states
        for timestep in range(x.shape[1]):
            output, next_states, _ = self._step(x[:, timestep], next_states)
            outputs.append(output)
        return torch.stack(outputs, dim=1), next_states

    def hidden_sequence(
        self,
        V: torch.Tensor,
        t: torch.Tensor,
        states: Optional[list] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        """Return predictions and final-layer hidden values for RRAD."""
        _validate_time(V, t, self.input_dim)
        x = torch.cat((V, t), dim=-1)
        outputs = []
        hidden_values = []
        next_states = states
        for timestep in range(x.shape[1]):
            output, next_states, hidden = self._step(x[:, timestep], next_states)
            outputs.append(output)
            hidden_values.append(hidden)
        return (
            torch.stack(outputs, dim=1),
            torch.stack(hidden_values, dim=1),
            next_states,
        )

    def step(
        self,
        V_t: torch.Tensor,
        t_t: torch.Tensor,
        states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Advance one physical timestep for streaming inference only."""
        if V_t.ndim != 2 or t_t.ndim != 2 or t_t.shape[-1] != 1:
            raise ValueError("step inputs must be [batch, features]")
        if V_t.shape[0] != t_t.shape[0] or V_t.shape[-1] + 1 != self.input_dim:
            raise ValueError("invalid streaming feature dimensions")
        output, next_states, _ = self._step(torch.cat((V_t, t_t), dim=-1), states)
        return output, next_states

    def compute_physics_loss(
        self,
        V: torch.Tensor,
        t: torch.Tensor,
        I_pred: torch.Tensor,
        I_true: torch.Tensor,
        vteam_params: dict,
        lambda_data: float = 1.0,
        lambda_pde: float = 0.1,
        lambda_ic: float = 0.1,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute sequence-weighted data, temporal, and initial losses."""
        del vteam_params
        _validate_time(V, t, self.input_dim)
        require_sequence_tensor(I_pred, name="I_pred", feature_size=self.output_dim)
        require_sequence_tensor(I_true, name="I_true", feature_size=self.output_dim)
        if I_pred.shape != I_true.shape or I_pred.shape[:2] != V.shape[:2]:
            raise ValueError("prediction and target shapes must match the input sequence")
        loss_data = torch.mean((I_pred - I_true) ** 2)
        if t.requires_grad:
            derivative = torch.autograd.grad(
                I_pred.sum(), t, create_graph=True, retain_graph=True
            )[0]
            maximum_derivative = 1e6
            loss_pde = torch.mean(
                torch.relu(torch.abs(derivative) - maximum_derivative) ** 2
            )
        else:
            loss_pde = I_pred.new_zeros(())
        loss_ic = torch.mean((I_pred[:, 0] - I_true[:, 0]) ** 2)
        total = lambda_data * loss_data + lambda_pde * loss_pde + lambda_ic * loss_ic
        return total, {
            "total": float(total.detach()),
            "data": float(loss_data.detach()),
            "pde": float(loss_pde.detach()),
            "ic": float(loss_ic.detach()),
        }


if __name__ == "__main__":
    model = xLSTMTeacher(hidden_size=16, num_layers=2, num_heads=4)
    voltage = torch.randn(2, 8, 1)
    time = torch.arange(8, dtype=torch.float32).view(1, 8, 1).repeat(2, 1, 1)
    prediction, final_state = model(voltage, time)
    print(tuple(prediction.shape), len(final_state))
