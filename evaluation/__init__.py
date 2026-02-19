"""Evaluation module for Ψ-xLSTM"""
from .metrics import (
    compute_spectral_accuracy,
    compute_compression_metrics,
    benchmark_inference_speed,
    compute_all_metrics
)

__all__ = [
    'compute_spectral_accuracy',
    'compute_compression_metrics',
    'benchmark_inference_speed',
    'compute_all_metrics'
]