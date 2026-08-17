"""Chronological Ψ-xLSTM student with clustered forget-gate dynamics."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn

from models.xlstm_teacher import _validate_time


class TimeConstantClusteringLSTM(nn.Module):
    """One internal recurrent timestep with time-constant regularization."""

    def __init__(self, input_size: int, hidden_size: int, num_clusters: int = 3):
        super().__init__()
        if input_size < 1 or hidden_size < 1:
            raise ValueError("cell dimensions must be positive")
        if num_clusters < 1 or num_clusters > hidden_size:
            raise ValueError("num_clusters must be between one and hidden_size")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_clusters = num_clusters
        self.W_gates = nn.Linear(input_size, 4 * hidden_size)
        self.R_gates = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        nn.init.zeros_(self.R_gates.weight)
        self.cluster_centers = nn.Parameter(
            torch.linspace(-2.0, 2.0, num_clusters).unsqueeze(1).repeat(1, input_size)
            + 0.1 * torch.randn(num_clusters, input_size)
        )
        self.register_buffer(
            "cluster_assignments", torch.arange(hidden_size) % num_clusters
        )
        with torch.no_grad():
            self.W_gates.bias[hidden_size : 2 * hidden_size] = torch.linspace(
                2.0, 7.0, hidden_size
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
                raise ValueError("invalid clustered-LSTM state shape")
        gates = self.W_gates(x) + self.R_gates(hidden)
        input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
        input_gate = torch.exp(input_gate.clamp(max=10.0))
        forget_gate = torch.sigmoid(forget_gate)
        candidate = torch.tanh(candidate)
        output_gate = torch.sigmoid(output_gate)
        next_memory = forget_gate * memory + input_gate * candidate
        next_hidden = output_gate * torch.tanh(next_memory)
        return next_hidden, (next_hidden, next_memory)

    def compute_clustering_loss(self) -> torch.Tensor:
        """Distance from each forget-gate row to its nearest cluster center."""
        forget_weights = self.W_gates.weight[
            self.hidden_size : 2 * self.hidden_size
        ]
        distances = torch.cdist(forget_weights, self.cluster_centers).square()
        return distances.min(dim=1).values.mean()

    def update_cluster_assignments(self) -> None:
        """Update assignments using only parameter values, never sequence samples."""
        with torch.no_grad():
            forget_weights = self.W_gates.weight[
                self.hidden_size : 2 * self.hidden_size
            ]
            kmeans = KMeans(
                n_clusters=self.num_clusters,
                init="k-means++",
                random_state=42,
                n_init=20,
                max_iter=300,
            )
            labels = kmeans.fit_predict(forget_weights.detach().cpu().numpy())
            self.cluster_assignments.copy_(
                torch.as_tensor(labels, device=self.cluster_assignments.device)
            )
            self.cluster_centers.copy_(
                torch.as_tensor(
                    kmeans.cluster_centers_,
                    device=self.cluster_centers.device,
                    dtype=self.cluster_centers.dtype,
                )
            )

    def get_discovered_time_constants(self) -> np.ndarray:
        with torch.no_grad():
            return torch.exp(-self.cluster_centers.mean(dim=1)).cpu().numpy()


class ClusteringStudent(nn.Module):
    """Sequence-only clustered recurrent Ψ-xLSTM student."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_size: int = 32,
        num_layers: int = 2,
        output_dim: int = 1,
        num_clusters: int = 3,
    ):
        super().__init__()
        if input_dim < 2 or hidden_size < 1 or num_layers < 1 or output_dim < 1:
            raise ValueError("model dimensions are invalid")
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.num_clusters = num_clusters
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.lstm_cells = nn.ModuleList(
            TimeConstantClusteringLSTM(hidden_size, hidden_size, num_clusters)
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
        """Advance one physical sample for streaming inference only."""
        if V_t.ndim != 2 or t_t.ndim != 2 or t_t.shape[-1] != 1:
            raise ValueError("step inputs must be [batch, features]")
        if V_t.shape[0] != t_t.shape[0] or V_t.shape[-1] + 1 != self.input_dim:
            raise ValueError("invalid streaming feature dimensions")
        output, next_states, _ = self._step(torch.cat((V_t, t_t), dim=-1), states)
        return output, next_states

    def compute_total_clustering_loss(self) -> torch.Tensor:
        return torch.stack(
            [cell.compute_clustering_loss() for cell in self.lstm_cells]
        ).mean()

    def update_all_clusters(self) -> None:
        for cell in self.lstm_cells:
            cell.update_cluster_assignments()

    def get_all_time_constants(self) -> dict[str, np.ndarray]:
        return {
            f"layer_{index}": cell.get_discovered_time_constants()
            for index, cell in enumerate(self.lstm_cells)
        }

    def count_parameters(self) -> Tuple[int, int]:
        original = sum(parameter.numel() for parameter in self.parameters())
        compressed = original
        for cell in self.lstm_cells:
            full = cell.hidden_size * cell.input_size
            tied = cell.num_clusters * cell.input_size
            compressed -= full - tied
        return original, compressed


if __name__ == "__main__":
    student = ClusteringStudent(hidden_size=16, num_layers=2, num_clusters=3)
    voltage = torch.randn(2, 8, 1)
    time = torch.arange(8, dtype=torch.float32).view(1, 8, 1).repeat(2, 1, 1)
    prediction, states = student(voltage, time)
    print(tuple(prediction.shape), len(states), student.count_parameters())
