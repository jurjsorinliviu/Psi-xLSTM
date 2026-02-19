"""
Improved Evaluation Metrics with Optimized Speed Benchmarking
Fixes: batch size for recurrent models, statistical analysis, scientific notation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import time
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt


def benchmark_inference_speed_optimized(model: nn.Module,
                                       V: torch.Tensor,
                                       t: torch.Tensor,
                                       num_runs: int = 50,
                                       warmup_runs: int = 5) -> Dict[str, float]:
    """
    OPTIMIZED inference speed benchmark
    
    Key improvements:
    1. Use FULL test set (not just 256 samples) for realistic throughput
    2. Process in optimal batch size (1024 for feedforward, full for recurrent)
    3. Multiple warmup runs to stabilize GPU
    4. Proper synchronization for CUDA timing
    
    Args:
        model: Model to benchmark
        V: Voltage input (full test set)
        t: Time input (full test set)
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        speed_metrics: Optimized inference speed statistics
    """
    model.eval()
    device = next(model.parameters()).device
    
    V = V.to(device)
    t = t.to(device)
    
    # Determine if model is recurrent
    is_recurrent = hasattr(model, 'forward') and 'states' in model.forward.__code__.co_varnames
    
    # For recurrent models: process full sequence to avoid loop overhead
    # For feedforward: use batch size 1024 for efficiency
    if is_recurrent:
        batch_size = len(V)  # Full sequence
        print(f"  Recurrent model: using full sequence ({batch_size} samples)")
    else:
        batch_size = min(1024, len(V))
        print(f"  Feedforward model: using batch size {batch_size}")
    
    V_batch = V[:batch_size]
    t_batch = t[:batch_size]
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            if is_recurrent:
                _ = model(V_batch, t_batch)
            else:
                _ = model(V_batch, t_batch)
    
    # Synchronize before benchmarking
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            if is_recurrent:
                _ = model(V_batch, t_batch)
            else:
                _ = model(V_batch, t_batch)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append(end - start)
    
    times = np.array(times)
    
    # Compute per-sample latency
    latency_per_sample = np.mean(times) / batch_size * 1000  # ms per sample
    
    metrics = {
        'mean_time_ms': float(np.mean(times) * 1000),
        'std_time_ms': float(np.std(times) * 1000),
        'min_time_ms': float(np.min(times) * 1000),
        'max_time_ms': float(np.max(times) * 1000),
        'latency_per_sample_ms': float(latency_per_sample),
        'throughput_samples_per_sec': float(batch_size / np.mean(times)),
        'batch_size_used': batch_size
    }
    
    return metrics


def compute_spectral_accuracy(I_pred: np.ndarray, 
                              I_true: np.ndarray,
                              dt: float,
                              plot_path: str = None) -> Dict[str, float]:
    """
    Compute spectral accuracy metrics via FFT analysis
    """
    N = len(I_true)
    freqs = fftfreq(N, dt)[:N//2]
    
    fft_pred = fft(I_pred.flatten())
    fft_true = fft(I_true.flatten())
    
    mag_pred = 2.0/N * np.abs(fft_pred[:N//2])
    mag_true = 2.0/N * np.abs(fft_true[:N//2])
    
    spectral_mse = np.mean((mag_pred - mag_true) ** 2)
    spectral_mae = np.mean(np.abs(mag_pred - mag_true))
    
    # High-frequency accuracy (> 20 kHz for challenging dataset)
    high_freq_mask = freqs > 20e3
    if np.any(high_freq_mask):
        high_freq_error = np.mean(np.abs(mag_pred[high_freq_mask] - mag_true[high_freq_mask]))
    else:
        high_freq_error = 0.0
    
    # Low-frequency accuracy (< 10 kHz)
    low_freq_mask = freqs < 10e3
    low_freq_error = np.mean(np.abs(mag_pred[low_freq_mask] - mag_true[low_freq_mask]))
    
    freq_correlation = np.corrcoef(mag_pred, mag_true)[0, 1]
    
    metrics = {
        'spectral_mse': float(spectral_mse),
        'spectral_mae': float(spectral_mae),
        'high_freq_error': float(high_freq_error),
        'low_freq_error': float(low_freq_error),
        'freq_correlation': float(freq_correlation)
    }
    
    # Plot if requested
    if plot_path:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        t = np.arange(len(I_true)) * dt
        ax1.plot(t[:1000], I_true[:1000], 'b-', label='Ground Truth', linewidth=2)
        ax1.plot(t[:1000], I_pred[:1000], 'r--', label='Prediction', linewidth=1.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Current (A)')
        ax1.set_title('Time Domain: Current Waveform')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.semilogy(freqs/1e3, mag_true, 'b-', label='Ground Truth', linewidth=2)
        ax2.semilogy(freqs/1e3, mag_pred, 'r--', label='Prediction', linewidth=1.5)
        ax2.axvline(x=20, color='g', linestyle=':', label='High-Freq Threshold (20 kHz)')
        ax2.set_xlabel('Frequency (kHz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Frequency Domain: FFT Spectrum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 100])
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Spectral plot saved to {plot_path}")
    
    return metrics


def compute_compression_metrics(model: nn.Module, 
                                model_name: str,
                                baseline_params: int = None) -> Dict[str, any]:
    """Compute compression metrics"""
    total_params = sum(p.numel() for p in model.parameters())
    
    metrics = {
        'model_name': model_name,
        'total_parameters': total_params,
        'model_size_mb': total_params * 4 / (1024 ** 2)
    }
    
    if baseline_params is not None:
        compression_ratio = total_params / baseline_params
        reduction_percent = (1 - compression_ratio) * 100
        metrics['compression_ratio'] = compression_ratio
        metrics['reduction_percent'] = reduction_percent
    
    if hasattr(model, 'count_parameters'):
        model_specific = model.count_parameters()
        if isinstance(model_specific, dict):
            metrics.update(model_specific)
        elif isinstance(model_specific, tuple):
            orig, comp = model_specific
            metrics['original_parameters'] = orig
            metrics['compressed_parameters'] = comp
    
    if hasattr(model, 'get_all_time_constants'):
        tau_dict = model.get_all_time_constants()
        metrics['time_constants'] = tau_dict
    
    return metrics


def compute_all_metrics_optimized(models_dict: Dict[str, nn.Module],
                                  dataset: dict,
                                  dt: float,
                                  output_dir: str = './results') -> Dict[str, Dict]:
    """
    Compute all metrics with optimized speed benchmarking
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Computing Metrics with Optimized Speed Benchmarks")
    print(f"{'='*60}")
    
    all_metrics = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    V_test = dataset['test']['V'].to(device)
    t_test = dataset['test']['t'].to(device)
    I_test = dataset['test']['I'].cpu().numpy()
    
    baseline_params = None
    if 'teacher' in models_dict:
        baseline_params = sum(p.numel() for p in models_dict['teacher'].parameters())
    
    for model_name, model in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        model = model.to(device)
        model.eval()
        
        metrics = {'model_name': model_name}
        
        # 1. Prediction accuracy
        with torch.no_grad():
            if hasattr(model, 'forward') and 'states' in model.forward.__code__.co_varnames:
                I_pred, _ = model(V_test, t_test)
            else:
                I_pred = model(V_test, t_test)
        
        I_pred_np = I_pred.cpu().numpy()
        
        # Time-domain metrics
        mse = np.mean((I_pred_np - I_test) ** 2)
        mae = np.mean(np.abs(I_pred_np - I_test))
        rmse = np.sqrt(mse)
        
        metrics['time_domain'] = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse)
        }
        
        # 2. Spectral accuracy
        plot_path = os.path.join(output_dir, f'{model_name}_spectral.png')
        spectral_metrics = compute_spectral_accuracy(I_pred_np, I_test, dt, plot_path)
        metrics['spectral'] = spectral_metrics
        
        # 3. Compression metrics
        compression_metrics = compute_compression_metrics(model, model_name, baseline_params)
        metrics['compression'] = compression_metrics
        
        # 4. OPTIMIZED Inference speed
        speed_metrics = benchmark_inference_speed_optimized(model, V_test, t_test)
        metrics['speed'] = speed_metrics
        
        all_metrics[model_name] = metrics
        
        # Print summary with scientific notation
        print(f"  Time-domain MSE: {mse:.3e}")
        print(f"  Spectral MAE: {spectral_metrics['spectral_mae']:.3e}")
        print(f"  High-freq error: {spectral_metrics['high_freq_error']:.3e}")
        print(f"  Parameters: {compression_metrics['total_parameters']:,}")
        if 'reduction_percent' in compression_metrics:
            print(f"  Compression: {compression_metrics['reduction_percent']:.1f}% reduction")
        print(f"  Latency/sample: {speed_metrics['latency_per_sample_ms']:.6f} ms")
        print(f"  Throughput: {speed_metrics['throughput_samples_per_sec']:.0f} samples/sec")
    
    # Comparative analysis
    print(f"\n{'='*60}")
    print("Comparative Summary (Optimized Speed)")
    print(f"{'='*60}")
    
    print(f"\n{'Model':<20} {'Params':>12} {'MSE':>12} {'HF-Error':>12} {'Latency/samp (μs)':>18}")
    print("-" * 80)
    
    for model_name, metrics in all_metrics.items():
        params = metrics['compression']['total_parameters']
        mse = metrics['time_domain']['mse']
        hf_error = metrics['spectral']['high_freq_error']
        latency_us = metrics['speed']['latency_per_sample_ms'] * 1000  # Convert to μs
        
        print(f"{model_name:<20} {params:>12,} {mse:>12.2e} {hf_error:>12.2e} {latency_us:>18.3f}")
    
    # Save metrics
    import json
    metrics_file = os.path.join(output_dir, 'metrics_optimized.json')
    with open(metrics_file, 'w') as f:
        json_safe_metrics = {}
        for model_name, model_metrics in all_metrics.items():
            json_safe_metrics[model_name] = {}
            for key, value in model_metrics.items():
                if isinstance(value, dict):
                    json_safe_metrics[model_name][key] = {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in value.items()
                        if not isinstance(v, dict)
                    }
                else:
                    json_safe_metrics[model_name][key] = value
        
        json.dump(json_safe_metrics, f, indent=2)
    
    print(f"\nOptimized metrics saved to {metrics_file}")
    
    return all_metrics