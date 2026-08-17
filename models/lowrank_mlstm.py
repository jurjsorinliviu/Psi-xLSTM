"""Chronological low-rank matrix-memory student for Psi-xLSTM."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from models.xlstm_teacher import _validate_time


class LowRankMLSTMCell(nn.Module):
    """One internal low-rank matrix-memory timestep."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rank: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        if min(input_size, hidden_size, rank, num_heads) < 1:
            raise ValueError("cell dimensions and rank must be positive")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if rank > self.head_dim:
            raise ValueError("rank cannot exceed the per-head hidden dimension")

        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        self.U = nn.Parameter(
            torch.randn(num_heads, self.head_dim, rank) * 0.01
        )
        self.V = nn.Parameter(
            torch.randn(num_heads, self.head_dim, rank) * 0.01
        )
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
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance one timestep; public training calls the sequence wrapper."""
        if x.ndim != 2 or x.shape[-1] != self.input_size:
            raise ValueError(
                f"cell input must have shape [batch, {self.input_size}]"
            )
        batch_size = x.shape[0]
        q = self.W_q(x).view(batch_size, self.num_heads, self.head_dim)
        k = self.W_k(x).view(batch_size, self.num_heads, self.head_dim)
        v = self.W_v(x).view(batch_size, self.num_heads, self.head_dim)
        input_gate = (
            torch.exp(self.igate(x).clamp(max=10.0)).unsqueeze(-1).unsqueeze(-1)
        )
        forget_gate = torch.sigmoid(self.fgate(x)).unsqueeze(-1).unsqueeze(-1)
        if state is None:
            compressed = x.new_zeros(
                batch_size, self.num_heads, self.rank, self.rank
            )
        else:
            expected = (batch_size, self.num_heads, self.rank, self.rank)
            if state.shape != expected:
                raise ValueError("invalid low-rank matrix-state shape")
            compressed = state

        value_compressed = torch.einsum("ndr,bnd->bnr", self.U, v)
        key_compressed = torch.einsum("ndr,bnd->bnr", self.V, k)
        next_compressed = forget_gate * compressed + input_gate * (
            value_compressed.unsqueeze(-1) @ key_compressed.unsqueeze(-2)
        )
        query_compressed = torch.einsum("ndr,bnd->bnr", self.U, q)
        hidden_compressed = torch.matmul(
            next_compressed, query_compressed.unsqueeze(-1)
        ).squeeze(-1)
        hidden = torch.einsum("ndr,bnr->bnd", self.U, hidden_compressed)
        hidden = self.out_norm(hidden.reshape(batch_size, self.hidden_size))
        return hidden, next_compressed

    def get_compression_ratio(self) -> float:
        full_size = self.num_heads * self.head_dim**2
        compressed_size = self.num_heads * (
            2 * self.head_dim * self.rank + self.rank**2
        )
        return compressed_size / full_size


class LowRankMLSTM(nn.Module):
    """Sequence-only low-rank recurrent Psi-xLSTM student."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_size: int = 32,
        num_layers: int = 2,
        output_dim: int = 1,
        rank: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        if min(input_dim, hidden_size, num_layers, output_dim, rank, num_heads) < 1:
            raise ValueError("model dimensions and rank must be positive")
        if input_dim < 2:
            raise ValueError("input_dim must include voltage and time")
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.rank = rank
        self.num_heads = num_heads
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.mlstm_cells = nn.ModuleList(
            LowRankMLSTMCell(hidden_size, hidden_size, rank, num_heads)
            for _ in range(num_layers)
        )
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def _step(self, x_t: torch.Tensor, states: list | None):
        if states is None:
            states = [None] * self.num_layers
        if len(states) != self.num_layers:
            raise ValueError(f"states must contain {self.num_layers} layer states")
        hidden = self.input_proj(x_t)
        next_states = []
        for cell, state in zip(self.mlstm_cells, states):
            hidden, next_state = cell(hidden, state)
            next_states.append(next_state)
        return self.output_proj(hidden), next_states, hidden

    def forward(
        self,
        V: torch.Tensor,
        t: torch.Tensor,
        states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Process one or more complete chronological trajectories."""
        _validate_time(V, t, self.input_dim)
        values = torch.cat((V, t), dim=-1)
        outputs = []
        next_states = states
        for timestep in range(values.shape[1]):
            output, next_states, _ = self._step(values[:, timestep], next_states)
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
        values = torch.cat((V, t), dim=-1)
        outputs = []
        hidden_values = []
        next_states = states
        for timestep in range(values.shape[1]):
            output, next_states, hidden = self._step(values[:, timestep], next_states)
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
        """Advance one sample for explicitly stateful streaming inference only."""
        if V_t.ndim != 2 or t_t.ndim != 2 or t_t.shape[-1] != 1:
            raise ValueError("step inputs must be [batch, features]")
        if V_t.shape[0] != t_t.shape[0] or V_t.shape[-1] + 1 != self.input_dim:
            raise ValueError("invalid streaming feature dimensions")
        output, next_states, _ = self._step(torch.cat((V_t, t_t), dim=-1), states)
        return output, next_states

    def count_parameters(self) -> dict[str, float | int]:
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        full_rank = sum(
            cell.num_heads * cell.head_dim**2 for cell in self.mlstm_cells
        )
        compressed = sum(
            2 * cell.num_heads * cell.head_dim * cell.rank
            for cell in self.mlstm_cells
        )
        ratio = compressed / full_rank
        return {
            "total_parameters": total_parameters,
            "matrix_memory_full": full_rank,
            "matrix_memory_compressed": compressed,
            "compression_ratio": ratio,
            "compression_percentage": (1.0 - ratio) * 100.0,
        }

    def get_eigenmode_analysis(self) -> dict[str, list[dict[str, object]]]:
        analysis: dict[str, list[dict[str, object]]] = {}
        for layer_index, cell in enumerate(self.mlstm_cells):
            heads = []
            for head_index in range(cell.num_heads):
                effective = (
                    cell.U[head_index].detach().cpu()
                    @ cell.V[head_index].detach().cpu().T
                )
                singular_values = torch.linalg.svdvals(effective).numpy()
                heads.append(
                    {"singular_values": singular_values, "rank": cell.rank}
                )
            analysis[f"layer_{layer_index}"] = heads
        return analysis


if __name__ == "__main__":
    model = LowRankMLSTM(hidden_size=16, num_layers=2, rank=2, num_heads=4)
    voltage = torch.randn(2, 8, 1)
    time = torch.arange(8, dtype=torch.float32).view(1, 8, 1).repeat(2, 1, 1)
    prediction, final_states = model(voltage, time)
    print(tuple(prediction.shape), len(final_states), model.count_parameters())
