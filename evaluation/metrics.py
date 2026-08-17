"""Evaluation that preserves trajectory boundaries and chronological order."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fft import fft, fftfreq
from torch import nn

from data.memristor_generator import require_sequence_tensor, validate_dataset


def _trajectory_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[0] < 1 or array.shape[1] <= 1:
        raise ValueError(
            f"{name} must have shape [trajectory, sequence_length, outputs] "
            "with sequence_length > 1"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def compute_spectral_accuracy(
    I_pred: np.ndarray,
    I_true: np.ndarray,
    dt: float,
    plot_path: str | None = None,
    high_frequency_hz: float = 10e3,
    low_frequency_hz: float = 5e3,
) -> Dict[str, float]:
    """Average per-trajectory spectra without joining unrelated sources."""
    prediction = _trajectory_array(I_pred, "I_pred")
    target = _trajectory_array(I_true, "I_true")
    if prediction.shape != target.shape:
        raise ValueError("I_pred and I_true must have identical shapes")
    if dt <= 0:
        raise ValueError("dt must be positive")
    sequence_length = target.shape[1]
    frequencies = fftfreq(sequence_length, dt)[: sequence_length // 2]
    predicted_spectra = np.abs(fft(prediction, axis=1)[:, : sequence_length // 2])
    target_spectra = np.abs(fft(target, axis=1)[:, : sequence_length // 2])
    predicted_spectra *= 2.0 / sequence_length
    target_spectra *= 2.0 / sequence_length
    difference = predicted_spectra - target_spectra
    high_mask = frequencies > high_frequency_hz
    low_mask = frequencies < low_frequency_hz
    high_error = float(np.mean(np.abs(difference[:, high_mask]))) if high_mask.any() else 0.0
    low_error = float(np.mean(np.abs(difference[:, low_mask]))) if low_mask.any() else 0.0
    correlations = []
    for trajectory in range(target.shape[0]):
        for output in range(target.shape[2]):
            left = predicted_spectra[trajectory, :, output]
            right = target_spectra[trajectory, :, output]
            if np.std(left) == 0 or np.std(right) == 0:
                correlations.append(1.0 if np.allclose(left, right) else 0.0)
            else:
                correlations.append(float(np.corrcoef(left, right)[0, 1]))
    metrics = {
        "spectral_mse": float(np.mean(difference**2)),
        "spectral_mae": float(np.mean(np.abs(difference))),
        "high_freq_error": high_error,
        "low_freq_error": low_error,
        "freq_correlation": float(np.mean(correlations)),
    }
    if plot_path:
        os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
        time_axis = np.arange(sequence_length) * dt
        figure, (time_axis_plot, frequency_plot) = plt.subplots(2, 1, figsize=(12, 8))
        time_axis_plot.plot(time_axis, target[0, :, 0], label="Ground truth")
        time_axis_plot.plot(time_axis, prediction[0, :, 0], "--", label="Prediction")
        time_axis_plot.set(xlabel="Time (s)", ylabel="Current", title="First held-out trajectory")
        time_axis_plot.legend()
        time_axis_plot.grid(True, alpha=0.3)
        frequency_plot.semilogy(
            frequencies / 1e3, target_spectra[0, :, 0], label="Ground truth"
        )
        frequency_plot.semilogy(
            frequencies / 1e3,
            predicted_spectra[0, :, 0],
            "--",
            label="Prediction",
        )
        frequency_plot.set(xlabel="Frequency (kHz)", ylabel="Magnitude", title="Spectrum")
        frequency_plot.legend()
        frequency_plot.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
    return metrics


def compute_compression_metrics(
    model: nn.Module,
    model_name: str,
    baseline_params: int | None = None,
) -> Dict[str, Any]:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "total_parameters": total_parameters,
        "model_size_mb": total_parameters * 4 / 1024**2,
    }
    if baseline_params is not None:
        metrics["compression_ratio"] = total_parameters / baseline_params
        metrics["reduction_percent"] = (
            1.0 - total_parameters / baseline_params
        ) * 100.0
    if hasattr(model, "count_parameters"):
        details = model.count_parameters()
        if isinstance(details, dict):
            metrics.update(details)
        elif isinstance(details, tuple):
            metrics["original_parameters"], metrics["compressed_parameters"] = details
    if hasattr(model, "get_all_time_constants"):
        metrics["time_constants"] = model.get_all_time_constants()
    if hasattr(model, "get_eigenmode_analysis"):
        metrics["eigenmodes"] = model.get_eigenmode_analysis()
    return metrics


def _predict(model: nn.Module, V: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    result = model(V, t)
    return result[0] if isinstance(result, tuple) else result


def benchmark_inference_speed(
    model: nn.Module,
    V: torch.Tensor,
    t: torch.Tensor,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """Benchmark full-trajectory inference with state reset on every run."""
    require_sequence_tensor(V, name="V")
    require_sequence_tensor(t, name="t", feature_size=1)
    if V.shape[:2] != t.shape[:2]:
        raise ValueError("V and t must share batch and sequence dimensions")
    if num_runs < 1 or warmup_runs < 0:
        raise ValueError("invalid benchmark run counts")
    model.eval()
    device = next(model.parameters()).device
    V = V.to(device)
    t = t.to(device)
    with torch.no_grad():
        for _ in range(warmup_runs):
            _predict(model, V, t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _predict(model, V, t)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
    values = np.asarray(elapsed)
    timesteps = V.shape[0] * V.shape[1]
    return {
        "mean_time_ms": float(values.mean() * 1000),
        "std_time_ms": float(values.std() * 1000),
        "min_time_ms": float(values.min() * 1000),
        "max_time_ms": float(values.max() * 1000),
        "latency_per_timestep_ms": float(values.mean() * 1000 / timesteps),
        "throughput_timesteps_per_sec": float(timesteps / values.mean()),
        "trajectories": int(V.shape[0]),
        "sequence_length": int(V.shape[1]),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def compute_all_metrics(
    models_dict: Dict[str, nn.Module],
    dataset: dict,
    dt: float,
    output_dir: str = "./results",
    *,
    device: str | None = None,
    benchmark_runs: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate every model on the same intact held-out trajectories."""
    validate_dataset(dataset)
    os.makedirs(output_dir, exist_ok=True)
    target_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    voltage = dataset["test"]["V"].to(target_device)
    time_values = dataset["test"]["t"].to(target_device)
    target = dataset["test"]["I"].to(target_device)
    teacher_parameters = None
    if "teacher" in models_dict:
        teacher_parameters = sum(
            parameter.numel() for parameter in models_dict["teacher"].parameters()
        )
    all_metrics: Dict[str, Dict[str, Any]] = {}
    for model_name, model in models_dict.items():
        model = model.to(target_device).eval()
        with torch.no_grad():
            prediction = _predict(model, voltage, time_values)
        if prediction.shape != target.shape:
            raise ValueError(
                f"{model_name} returned {tuple(prediction.shape)}, expected "
                f"{tuple(target.shape)}"
            )
        error = prediction - target
        mse = float(torch.sum(error.square()) / target.numel())
        mae = float(torch.sum(error.abs()) / target.numel())
        all_metrics[model_name] = {
            "model_name": model_name,
            "time_domain": {"mse": mse, "mae": mae, "rmse": mse**0.5},
            "spectral": compute_spectral_accuracy(
                prediction.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                dt,
                os.path.join(output_dir, f"{model_name}_spectral.png"),
            ),
            "compression": compute_compression_metrics(
                model, model_name, teacher_parameters
            ),
            "speed": benchmark_inference_speed(
                model,
                voltage,
                time_values,
                num_runs=benchmark_runs,
                warmup_runs=min(5, benchmark_runs),
            ),
        }
    with open(
        os.path.join(output_dir, "chapter4_metrics.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(_json_safe(all_metrics), handle, indent=2)
    return all_metrics


if __name__ == "__main__":
    samples = np.sin(np.linspace(0, 4 * np.pi, 32))[None, :, None]
    print(compute_spectral_accuracy(samples, samples, 1e-4))
