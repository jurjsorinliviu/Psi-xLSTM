"""Sequence-safe recurrent relation-aware distillation (RRAD)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from data.memristor_generator import require_sequence_tensor
from models.xlstm_teacher import _validate_time


class RecurrentRelationAwareDistillation(nn.Module):
    """Distill chronological teacher trajectories into a recurrent student.

    Both networks receive the same ordered sequence. Recurrent states may be
    supplied only to continue the immediately preceding TBPTT chunk and the
    caller receives the updated states for the next contiguous chunk.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.1,
    ):
        super().__init__()
        if min(alpha, beta, gamma) < 0:
            raise ValueError("RRAD weights must be nonnegative")
        if not hasattr(teacher, "hidden_sequence") or not hasattr(
            student, "hidden_sequence"
        ):
            raise TypeError("teacher and student must expose hidden_sequence()")
        self.teacher = teacher
        self.student = student
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        teacher_size = int(teacher.hidden_size)
        student_size = int(student.hidden_size)
        self.hidden_projection: nn.Module
        if teacher_size == student_size:
            self.hidden_projection = nn.Identity()
        else:
            self.hidden_projection = nn.Linear(student_size, teacher_size, bias=False)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

    def compute_distillation_loss(
        self,
        V: torch.Tensor,
        t: torch.Tensor,
        I_true: torch.Tensor,
        student_states: Optional[list] = None,
        teacher_states: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float], list, list]:
        """Compute sequence-level RRAD losses and return propagated states."""
        _validate_time(V, t, int(self.student.input_dim))
        require_sequence_tensor(
            I_true, name="I_true", feature_size=int(self.student.output_dim)
        )
        if I_true.shape[:2] != V.shape[:2]:
            raise ValueError("I_true must share the input batch and sequence dimensions")
        if not t.requires_grad:
            raise ValueError("t must require gradients for RRAD temporal matching")

        teacher_prediction, teacher_hidden, next_teacher_states = (
            self.teacher.hidden_sequence(V, t, teacher_states)
        )
        teacher_derivative = torch.autograd.grad(
            teacher_prediction.sum(),
            t,
            create_graph=False,
            retain_graph=True,
        )[0].detach()
        student_prediction, student_hidden, next_student_states = (
            self.student.hidden_sequence(V, t, student_states)
        )
        student_derivative = torch.autograd.grad(
            student_prediction.sum(),
            t,
            create_graph=True,
            retain_graph=True,
        )[0]

        loss_data = torch.mean((student_prediction - I_true) ** 2)
        loss_output = torch.mean(
            (student_prediction - teacher_prediction.detach()) ** 2
        )
        projected_hidden = self.hidden_projection(student_hidden)
        loss_hidden = torch.mean((projected_hidden - teacher_hidden.detach()) ** 2)
        loss_gradient = torch.mean((student_derivative - teacher_derivative) ** 2)
        loss_structure = self._compute_structure_loss()
        relation_loss = loss_output + loss_hidden
        total_loss = (
            loss_data
            + self.alpha * relation_loss
            + self.beta * loss_gradient
            + self.gamma * loss_structure
        )
        losses = {
            "total": float(total_loss.detach()),
            "data": float(loss_data.detach()),
            "output_matching": float(loss_output.detach()),
            "hidden_matching": float(loss_hidden.detach()),
            "gradient_matching": float(loss_gradient.detach()),
            "structure_discovery": float(loss_structure.detach()),
        }
        return total_loss, losses, next_student_states, next_teacher_states

    def _compute_structure_loss(self) -> torch.Tensor:
        if hasattr(self.student, "compute_total_clustering_loss"):
            return self.student.compute_total_clustering_loss()
        return next(self.student.parameters()).new_zeros(())

    def train_step(
        self,
        optimizer: torch.optim.Optimizer,
        V: torch.Tensor,
        t: torch.Tensor,
        I_true: torch.Tensor,
        student_states: Optional[list] = None,
        teacher_states: Optional[list] = None,
    ) -> Tuple[Dict[str, float], list, list]:
        optimizer.zero_grad(set_to_none=True)
        loss, losses, next_student, next_teacher = self.compute_distillation_loss(
            V,
            t,
            I_true,
            student_states=student_states,
            teacher_states=teacher_states,
        )
        loss.backward()
        trainable = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        return losses, next_student, next_teacher

    def update_student_structure(self, epoch: int, update_interval: int = 10) -> None:
        if update_interval < 1:
            raise ValueError("update_interval must be positive")
        if (epoch + 1) % update_interval == 0 and hasattr(
            self.student, "update_all_clusters"
        ):
            self.student.update_all_clusters()


class StandardPINNBaseline(nn.Module):
    """Static MLP baseline evaluated on intact sequence tensors."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_sizes: Optional[list[int]] = None,
        output_dim: int = 1,
    ):
        super().__init__()
        hidden_sizes = hidden_sizes or [64, 64, 64]
        if input_dim < 2 or output_dim < 1 or not hidden_sizes:
            raise ValueError("invalid baseline dimensions")
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers: list[nn.Module] = []
        current_size = input_dim
        for hidden_size in hidden_sizes:
            if hidden_size < 1:
                raise ValueError("hidden sizes must be positive")
            layers.extend((nn.Linear(current_size, hidden_size), nn.Tanh()))
            current_size = hidden_size
        layers.append(nn.Linear(current_size, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, V: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _validate_time(V, t, self.input_dim)
        return self.network(torch.cat((V, t), dim=-1))

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def create_baseline_pinn(
    input_dim: int = 2,
    hidden_size: int = 64,
    num_layers: int = 3,
    output_dim: int = 1,
) -> StandardPINNBaseline:
    return StandardPINNBaseline(
        input_dim=input_dim,
        hidden_sizes=[hidden_size] * num_layers,
        output_dim=output_dim,
    )


if __name__ == "__main__":
    from models.clustering_student import ClusteringStudent
    from models.xlstm_teacher import xLSTMTeacher

    teacher_model = xLSTMTeacher(hidden_size=16, num_layers=2, num_heads=4)
    student_model = ClusteringStudent(hidden_size=8, num_layers=2, num_clusters=3)
    objective = RecurrentRelationAwareDistillation(teacher_model, student_model)
    voltage = torch.randn(2, 8, 1)
    time = (
        torch.arange(8, dtype=torch.float32)
        .view(1, 8, 1)
        .repeat(2, 1, 1)
        .requires_grad_(True)
    )
    current = torch.randn(2, 8, 1)
    loss, components, _, _ = objective.compute_distillation_loss(
        voltage, time, current
    )
    print(float(loss.detach()), components)
