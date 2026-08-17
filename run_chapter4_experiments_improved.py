"""Train Psi-xLSTM models through the chronological sequence API.

Training is opt-in: pass ``--run-training``. Dataset items are complete source
trajectories, recurrent state resets between them, and TBPTT state is threaded
only through adjacent chunks from the same trajectory batch.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data.memristor_generator import (
    MemristorConfig,
    MemristorDataGenerator,
    validate_dataset,
)
from evaluation.metrics import compute_all_metrics
from models.clustering_student import ClusteringStudent
from models.lowrank_mlstm import LowRankMLSTM
from models.xlstm_teacher import xLSTMTeacher
from training.distillation import create_baseline_pinn
from training.trainer import train_student, train_teacher


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def train_static_baseline(
    model: torch.nn.Module,
    dataset: dict,
    *,
    epochs: int,
    trajectory_batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, list[float]]]:
    """Train the declared nonrecurrent control on intact sequence tensors."""
    validate_dataset(dataset)
    generator = torch.Generator().manual_seed(seed)
    trajectory_dataset = TensorDataset(
        dataset["train"]["V"],
        dataset["train"]["t"],
        dataset["train"]["I"],
    )
    loader = DataLoader(
        trajectory_dataset,
        batch_size=trajectory_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = {"train_loss": []}
    for _ in range(epochs):
        model.train()
        squared_error = 0.0
        observations = 0
        for voltage, time_values, current in loader:
            voltage = voltage.to(device)
            time_values = time_values.to(device)
            current = current.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(voltage, time_values)
            loss = torch.mean((prediction - current) ** 2)
            loss.backward()
            optimizer.step()
            squared_error += float(torch.sum((prediction.detach() - current) ** 2))
            observations += current.numel()
        history["train_loss"].append(squared_error / observations)
    return model, history


def train_one_seed(args: argparse.Namespace, seed: int, device: torch.device):
    set_seed(seed)
    seed_directory = Path(args.output_dir) / "seeds" / f"seed_{seed}"
    seed_directory.mkdir(parents=True, exist_ok=True)
    data_config = MemristorConfig(dt=args.dt, t_max=args.t_max)
    dataset = MemristorDataGenerator(data_config, seed=seed).generate_dataset(
        num_sequences=args.num_sequences,
        split_ratio=tuple(args.split_ratio),
        device=device,
        f_high_start=args.high_frequency_start,
        f_high_step=args.high_frequency_step,
        f_low_start=args.low_frequency_start,
        f_low_step=args.low_frequency_step,
        noise_level=args.noise_level,
    )
    validate_dataset(dataset)

    baseline = create_baseline_pinn(
        input_dim=2,
        hidden_size=args.baseline_hidden_size,
        num_layers=args.baseline_layers,
        output_dim=1,
    )
    baseline, baseline_history = train_static_baseline(
        baseline,
        dataset,
        epochs=args.baseline_epochs,
        trajectory_batch_size=args.trajectory_batch_size,
        learning_rate=args.learning_rate,
        device=device,
        seed=seed,
    )
    teacher = xLSTMTeacher(
        input_dim=2,
        hidden_size=args.teacher_hidden_size,
        num_layers=args.recurrent_layers,
        output_dim=1,
        use_mlstm=True,
        num_heads=args.num_heads,
    )
    teacher, teacher_history = train_teacher(
        teacher,
        dataset,
        num_epochs=args.teacher_epochs,
        batch_size=args.trajectory_batch_size,
        learning_rate=args.learning_rate,
        device=str(device),
        save_dir=str(seed_directory),
        chunk_length=args.chunk_length,
        seed=seed,
    )
    clustered = ClusteringStudent(
        input_dim=2,
        hidden_size=args.student_hidden_size,
        num_layers=args.recurrent_layers,
        output_dim=1,
        num_clusters=args.num_clusters,
    )
    clustered, clustered_history = train_student(
        clustered,
        teacher,
        dataset,
        num_epochs=args.student_epochs,
        batch_size=args.trajectory_batch_size,
        learning_rate=args.learning_rate,
        device=str(device),
        gamma=args.clustering_weight,
        save_dir=str(seed_directory),
        chunk_length=args.chunk_length,
        seed=seed,
        checkpoint_name="clustered_student_best.pth",
    )
    low_rank = LowRankMLSTM(
        input_dim=2,
        hidden_size=args.student_hidden_size,
        num_layers=args.recurrent_layers,
        output_dim=1,
        rank=args.rank,
        num_heads=args.num_heads,
    )
    low_rank, low_rank_history = train_student(
        low_rank,
        teacher,
        dataset,
        num_epochs=args.student_epochs,
        batch_size=args.trajectory_batch_size,
        learning_rate=args.learning_rate,
        device=str(device),
        gamma=0.0,
        save_dir=str(seed_directory),
        chunk_length=args.chunk_length,
        seed=seed,
        checkpoint_name="lowrank_student_best.pth",
    )
    models = {
        "baseline_pinn": baseline,
        "teacher": teacher,
        "psi_xlstm_clustering": clustered,
        "psi_xlstm_lowrank": low_rank,
    }
    for name, model in models.items():
        torch.save(model.state_dict(), seed_directory / f"{name}_final.pth")
    histories = {
        "baseline_pinn": baseline_history,
        "teacher": teacher_history,
        "psi_xlstm_clustering": clustered_history,
        "psi_xlstm_lowrank": low_rank_history,
    }
    with (seed_directory / "training_history.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_json_safe(histories), handle, indent=2)
    metrics = compute_all_metrics(
        models,
        dataset,
        data_config.dt,
        str(seed_directory / "evaluation"),
        device=str(device),
        benchmark_runs=args.benchmark_runs,
    )
    manifest = {
        "seed": seed,
        "device": str(device),
        "canonical_shape": "[batch, sequence_length, features]",
        "sequence_length": data_config.num_steps,
        "source_ids": {
            split: list(dataset[split]["source_ids"])
            for split in ("train", "val", "test")
        },
        "checkpoints": {
            name: str(seed_directory / f"{name}_final.pth") for name in models
        },
    }
    with (seed_directory / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chronological Psi-xLSTM teacher and student training"
    )
    parser.add_argument(
        "--run-training",
        action="store_true",
        help="required acknowledgement that model training should begin",
    )
    parser.add_argument("--output-dir", default="chapter4_results_sequence")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--num-sequences", type=int, default=10)
    parser.add_argument("--split-ratio", nargs=3, type=float, default=[0.7, 0.15, 0.15])
    parser.add_argument("--dt", type=float, default=5e-8)
    parser.add_argument("--t-max", type=float, default=2e-3)
    parser.add_argument("--noise-level", type=float, default=0.03)
    parser.add_argument("--high-frequency-start", type=float, default=50e3)
    parser.add_argument("--high-frequency-step", type=float, default=10e3)
    parser.add_argument("--low-frequency-start", type=float, default=1e3)
    parser.add_argument("--low-frequency-step", type=float, default=500.0)
    parser.add_argument("--baseline-epochs", type=int, default=50)
    parser.add_argument("--teacher-epochs", type=int, default=100)
    parser.add_argument("--student-epochs", type=int, default=150)
    parser.add_argument("--trajectory-batch-size", type=int, default=2)
    parser.add_argument("--chunk-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--baseline-hidden-size", type=int, default=64)
    parser.add_argument("--baseline-layers", type=int, default=3)
    parser.add_argument("--teacher-hidden-size", type=int, default=64)
    parser.add_argument("--student-hidden-size", type=int, default=32)
    parser.add_argument("--recurrent-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-clusters", type=int, default=3)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--clustering-weight", type=float, default=0.1)
    parser.add_argument("--benchmark-runs", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.run_training:
        print(
            "No training started. Re-run with --run-training after reviewing the "
            "trajectory, epoch, and device arguments."
        )
        return 0
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.chunk_length <= 1:
        raise ValueError("--chunk-length must be greater than one")
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    configuration.update(
        {
            "resolved_device": str(device),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_shape": "[batch, sequence_length, features]",
            "split_unit": "complete source trajectory",
        }
    )
    with (output_directory / "configuration.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(configuration, handle, indent=2)
    all_metrics = []
    for seed in args.seeds:
        all_metrics.append(train_one_seed(args, seed, device))
    with (output_directory / "all_seed_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_json_safe(all_metrics), handle, indent=2)
    print(f"Completed {len(args.seeds)} seed(s). Results: {output_directory.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
