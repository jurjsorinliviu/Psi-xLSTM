#!/usr/bin/env python3
"""Compare a chronological ngspice trace with a trained Psi-xLSTM model.

This utility does not call an analytical surrogate a circuit simulation. Supply
an ngspice/OpenVAF trace produced by the circuit implementation being tested.
The PyTorch model is reset once, then processes the full trace in time order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.clustering_student import ClusteringStudent


def _load_weights(model: torch.nn.Module, checkpoint: Path) -> None:
    try:
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        weights = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(weights)


def load_trace(
    path: Path,
    *,
    skip_rows: int,
    time_column: int,
    voltage_column: int,
    current_column: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.loadtxt(path, skiprows=skip_rows)
    if values.ndim != 2:
        raise ValueError("trace must be a two-dimensional numeric table")
    time_values = values[:, time_column]
    voltage = values[:, voltage_column]
    current = values[:, current_column]
    if len(time_values) <= 1 or np.any(np.diff(time_values) <= 0):
        raise ValueError("trace times must be strictly increasing")
    if not np.isfinite(np.column_stack((time_values, voltage, current))).all():
        raise ValueError("trace contains non-finite values")
    return time_values, voltage, current


def pytorch_prediction(
    checkpoint: Path,
    time_values: np.ndarray,
    voltage: np.ndarray,
    *,
    hidden_size: int,
    num_layers: int,
    num_clusters: int,
) -> np.ndarray:
    model = ClusteringStudent(
        input_dim=2,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_dim=1,
        num_clusters=num_clusters,
    )
    _load_weights(model, checkpoint)
    model.eval()
    voltage_tensor = torch.as_tensor(voltage, dtype=torch.float32)[None, :, None]
    time_tensor = torch.as_tensor(time_values, dtype=torch.float32)[None, :, None]
    with torch.no_grad():
        prediction, _ = model(voltage_tensor, time_tensor, states=None)
    return prediction[0, :, 0].cpu().numpy()


def comparison_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict:
    if reference.shape != prediction.shape:
        raise ValueError("reference and prediction lengths differ")
    error = prediction - reference
    mse = float(np.mean(error**2))
    denominator = float(np.ptp(reference))
    return {
        "mse": mse,
        "rmse": mse**0.5,
        "mae": float(np.mean(np.abs(error))),
        "correlation": float(np.corrcoef(reference, prediction)[0, 1]),
        "relative_mae_percent_of_range": (
            float(np.mean(np.abs(error))) / denominator * 100.0
            if denominator > 0
            else 0.0
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare an external circuit trace with chronological PyTorch inference"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("spice_comparison"))
    parser.add_argument("--skip-rows", type=int, default=1)
    parser.add_argument("--time-column", type=int, default=0)
    parser.add_argument("--voltage-column", type=int, default=1)
    parser.add_argument("--current-column", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-clusters", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    time_values, voltage, circuit_current = load_trace(
        args.trace,
        skip_rows=args.skip_rows,
        time_column=args.time_column,
        voltage_column=args.voltage_column,
        current_column=args.current_column,
    )
    prediction = pytorch_prediction(
        args.checkpoint,
        time_values,
        voltage,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_clusters=args.num_clusters,
    )
    metrics = comparison_metrics(circuit_current, prediction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(time_values, circuit_current, label="Circuit trace")
    axes[0].plot(time_values, prediction, "--", label="PyTorch sequence model")
    axes[0].set_ylabel("Current")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(time_values, prediction - circuit_current)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Prediction error")
    axes[1].grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.output_dir / "comparison.png", dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
