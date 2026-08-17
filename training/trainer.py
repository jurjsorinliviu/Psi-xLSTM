"""Chronological teacher and Psi-xLSTM student training loops."""

from __future__ import annotations

import os
import time
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data.memristor_generator import (
    contiguous_chunks,
    detach_state,
    validate_dataset,
)
from training.distillation import RecurrentRelationAwareDistillation


def _loader(
    split: dict,
    *,
    batch_size: int,
    shuffle_trajectories: bool,
    seed: int,
) -> DataLoader:
    """Batch complete trajectories; never expose timesteps as dataset items."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    trajectories = TensorDataset(split["V"], split["t"], split["I"])
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        trajectories,
        batch_size=batch_size,
        shuffle=shuffle_trajectories,
        generator=generator if shuffle_trajectories else None,
        num_workers=0,
    )


def _load_weights(model: nn.Module, path: str, device: torch.device) -> None:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)


def _validation_mse(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    squared_error = 0.0
    target_elements = 0
    model.eval()
    with torch.no_grad():
        for voltage, time_values, current in loader:
            voltage = voltage.to(device)
            time_values = time_values.to(device)
            current = current.to(device)
            prediction, _ = model(voltage, time_values, states=None)
            squared_error += float(torch.sum((prediction - current) ** 2))
            target_elements += current.numel()
    if target_elements == 0:
        raise ValueError("validation split is empty")
    return squared_error / target_elements


def train_teacher(
    teacher: nn.Module,
    dataset: dict,
    num_epochs: int = 100,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    lambda_data: float = 1.0,
    lambda_pde: float = 0.1,
    lambda_ic: float = 0.1,
    save_dir: str = ".",
    chunk_length: int | None = 256,
    seed: int = 42,
) -> Tuple[nn.Module, dict]:
    """Train on intact trajectories with correctly threaded TBPTT state."""
    validate_dataset(dataset)
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    target_device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)
    teacher = teacher.to(target_device)
    optimizer = torch.optim.AdamW(
        teacher.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    train_loader = _loader(
        dataset["train"],
        batch_size=batch_size,
        shuffle_trajectories=True,
        seed=seed,
    )
    validation_loader = _loader(
        dataset["val"],
        batch_size=batch_size,
        shuffle_trajectories=False,
        seed=seed,
    )
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_data_loss": [],
        "train_pde_loss": [],
        "train_ic_loss": [],
        "epoch_time": [],
    }
    best_validation = float("inf")
    checkpoint_path = os.path.join(save_dir, "teacher_best.pth")

    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        teacher.train()
        components = {"total": [], "data": [], "pde": [], "ic": []}
        progress = tqdm(
            train_loader,
            desc=f"teacher {epoch + 1}/{num_epochs}",
            leave=False,
        )
        for voltage, time_values, current in progress:
            voltage = voltage.to(target_device)
            time_values = time_values.to(target_device)
            current = current.to(target_device)
            recurrent_state = None
            for voltage_chunk, time_chunk, current_chunk in contiguous_chunks(
                voltage, time_values, current, chunk_length
            ):
                time_chunk = time_chunk.detach().clone().requires_grad_(True)
                optimizer.zero_grad(set_to_none=True)
                prediction, recurrent_state = teacher(
                    voltage_chunk, time_chunk, recurrent_state
                )
                loss, loss_values = teacher.compute_physics_loss(
                    voltage_chunk,
                    time_chunk,
                    prediction,
                    current_chunk,
                    {},
                    lambda_data,
                    lambda_pde,
                    lambda_ic,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
                optimizer.step()
                recurrent_state = detach_state(recurrent_state)
                for key in components:
                    components[key].append(loss_values[key])
                progress.set_postfix(loss=f"{loss_values['total']:.3e}")

        validation = _validation_mse(teacher, validation_loader, target_device)
        scheduler.step(validation)
        if validation < best_validation:
            best_validation = validation
            torch.save(teacher.state_dict(), checkpoint_path)
        history["train_loss"].append(float(np.mean(components["total"])))
        history["val_loss"].append(validation)
        history["train_data_loss"].append(float(np.mean(components["data"])))
        history["train_pde_loss"].append(float(np.mean(components["pde"])))
        history["train_ic_loss"].append(float(np.mean(components["ic"])))
        history["epoch_time"].append(time.perf_counter() - epoch_start)

    _load_weights(teacher, checkpoint_path, target_device)
    return teacher, history


def train_student(
    student: nn.Module,
    teacher: nn.Module,
    dataset: dict,
    num_epochs: int = 150,
    batch_size: int = 2,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
    update_clusters_interval: int = 10,
    save_dir: str = ".",
    chunk_length: int | None = 256,
    seed: int = 42,
    checkpoint_name: str | None = None,
) -> Tuple[nn.Module, dict]:
    """Train a recurrent student by RRAD on chronological TBPTT chunks."""
    validate_dataset(dataset)
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    target_device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(seed)
    student = student.to(target_device)
    teacher = teacher.to(target_device)
    teacher.eval()
    objective = RecurrentRelationAwareDistillation(
        teacher, student, alpha=alpha, beta=beta, gamma=gamma
    ).to(target_device)
    trainable = [
        parameter for parameter in objective.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    train_loader = _loader(
        dataset["train"],
        batch_size=batch_size,
        shuffle_trajectories=True,
        seed=seed,
    )
    validation_loader = _loader(
        dataset["val"],
        batch_size=batch_size,
        shuffle_trajectories=False,
        seed=seed,
    )
    history = {
        "train_loss": [],
        "val_loss": [],
        "output_matching_loss": [],
        "hidden_matching_loss": [],
        "gradient_matching_loss": [],
        "structure_loss": [],
        "epoch_time": [],
    }
    best_validation = float("inf")
    if checkpoint_name is None:
        checkpoint_name = f"{student.__class__.__name__.lower()}_best.pth"
    checkpoint_path = os.path.join(save_dir, checkpoint_name)

    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        student.train()
        teacher.eval()
        components = {
            "total": [],
            "data": [],
            "output_matching": [],
            "hidden_matching": [],
            "gradient_matching": [],
            "structure_discovery": [],
        }
        progress = tqdm(
            train_loader,
            desc=f"student {epoch + 1}/{num_epochs}",
            leave=False,
        )
        for voltage, time_values, current in progress:
            voltage = voltage.to(target_device)
            time_values = time_values.to(target_device)
            current = current.to(target_device)
            student_state = None
            teacher_state = None
            for voltage_chunk, time_chunk, current_chunk in contiguous_chunks(
                voltage, time_values, current, chunk_length
            ):
                time_chunk = time_chunk.detach().clone().requires_grad_(True)
                loss_values, student_state, teacher_state = objective.train_step(
                    optimizer,
                    voltage_chunk,
                    time_chunk,
                    current_chunk,
                    student_states=student_state,
                    teacher_states=teacher_state,
                )
                student_state = detach_state(student_state)
                teacher_state = detach_state(teacher_state)
                for key in components:
                    components[key].append(loss_values[key])
                progress.set_postfix(loss=f"{loss_values['total']:.3e}")

        objective.update_student_structure(epoch, update_clusters_interval)
        validation = _validation_mse(student, validation_loader, target_device)
        scheduler.step()
        if validation < best_validation:
            best_validation = validation
            torch.save(student.state_dict(), checkpoint_path)
        history["train_loss"].append(float(np.mean(components["total"])))
        history["val_loss"].append(validation)
        history["output_matching_loss"].append(
            float(np.mean(components["output_matching"]))
        )
        history["hidden_matching_loss"].append(
            float(np.mean(components["hidden_matching"]))
        )
        history["gradient_matching_loss"].append(
            float(np.mean(components["gradient_matching"]))
        )
        history["structure_loss"].append(
            float(np.mean(components["structure_discovery"]))
        )
        history["epoch_time"].append(time.perf_counter() - epoch_start)

    _load_weights(student, checkpoint_path, target_device)
    return student, history


if __name__ == "__main__":
    print("Use run_chapter4_experiments_improved.py to train chronological models.")
