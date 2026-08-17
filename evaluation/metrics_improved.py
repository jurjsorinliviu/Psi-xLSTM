"""Compatibility names for the canonical chronological evaluation module."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from evaluation.metrics import (
    benchmark_inference_speed,
    compute_all_metrics,
    compute_compression_metrics,
    compute_spectral_accuracy,
)


def benchmark_inference_speed_optimized(
    model: nn.Module,
    V: torch.Tensor,
    t: torch.Tensor,
    num_runs: int = 50,
    warmup_runs: int = 5,
) -> Dict[str, float]:
    """Benchmark intact sequences; the name is retained for old callers."""
    return benchmark_inference_speed(model, V, t, num_runs, warmup_runs)


def compute_all_metrics_optimized(
    models_dict: Dict[str, nn.Module],
    dataset: dict,
    dt: float,
    output_dir: str = "./results",
):
    """Delegate to the sole sequence-safe evaluation implementation."""
    return compute_all_metrics(models_dict, dataset, dt, output_dir)


__all__ = [
    "benchmark_inference_speed_optimized",
    "compute_all_metrics_optimized",
    "compute_compression_metrics",
    "compute_spectral_accuracy",
]
