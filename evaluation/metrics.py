"""
Evaluation Metrics for Chapter 4 Results
Includes spectral analysis, compression metrics, and speed benchmarks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import time
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt


def compute_spectral_accuracy(I_pred: np.ndarray, 
                              I_true: np.ndarray,
                              dt: float,
                              plot_path: str = None) -> Dict[str, float]:
    """
    Compute spectral accuracy metrics via FFT analysis
    Measures how well model captures high-frequency components
    
    Args:
        I_pred: Predicted current waveform
        I_true: Ground truth current waveform
        dt: Time step
        plot_path: Optional path to save FFT comparison plot
        
    Returns:
        metrics: Dictionary with spectral accuracy metrics
    """
    # Compute FFT
    N = len(I_true)
    freqs = fftfreq(N, dt)[:N//2]
    
    fft_pred = fft(I_pred.flatten())
    fft_true = fft(I_true.flatten())
    
    # Magnitude spectra
    mag_pred = 2.0/N * np.abs(fft_pred[:N//2])
    mag_true = 2.0/N * np.abs(fft_true[:N//2])
    
    # Spectral error
    spectral_mse = np.mean((mag_pred - mag_true) ** 2)
    spectral_mae = np.mean(np.abs(mag_pred - mag_true))
    
    # High-frequency accuracy (> 10 kHz)
    high_freq_mask = freqs > 10e3
    if np.any(high_freq_mask):
        high_freq_error = np.mean(np.abs(mag_pred[high_freq_mask] - mag_true[high_freq_mask]))
    else:
        high_freq_error = 0.0
    
    # Low-frequency accuracy (< 5 kHz)
    low_freq_mask = freqs < 5e3
    low_freq_error = np.mean(np.abs(mag_pred[low_freq_mask] - mag_true[low_freq_mask]))
    
    # Frequency correlation
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
        
        # Time domain
        t = np.arange(len(I_true)) * dt
        ax1.plot(t[:1000], I_true[:1000], 'b-', label='Ground Truth', linewidth=2)
        ax1.plot(t[:1000], I_pred[:1000], 'r--', label='Prediction', linewidth=1.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Current (A)')
        ax1.set_title('Time Domain: Current Waveform')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Frequency domain
        ax2.semilogy(freqs/1e3, mag_true, 'b-', label='Ground Truth', linewidth=2)
        ax2.semilogy(freqs/1e3, mag_pred, 'r--', label='Prediction', linewidth=1.5)
        ax2.axvline(x=10, color='g', linestyle=':', label='High-Freq Threshold (10 kHz)')
        ax2.set_xlabel('Frequency (kHz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Frequency Domain: FFT Spectrum')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 50])
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Spectral plot saved to {plot_path}")
    
    return metrics


def compute_compression_metrics(model: nn.Module, 
                               model_name: str,
                               baseline_params: int = None) -> Dict[str, any]:
    """
    Compute compression metrics
    
    Args:
        model: PyTorch model
        model_name: Name of model
        baseline_params: Optional baseline parameter count for comparison
        
    Returns:
        metrics: Compression statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    metrics = {
        'model_name': model_name,
        'total_parameters': total_params,
        'model_size_mb': total_params * 4 / (1024 ** 2)  # Assuming float32
    }
    
    if baseline_params is not None:
        compression_ratio = total_params / baseline_params
        reduction_percent = (1 - compression_ratio) * 100
        metrics['compression_ratio'] = compression_ratio
        metrics['reduction_percent'] = reduction_percent
    
    # Add model-specific metrics
    if hasattr(model, 'count_parameters'):
        model_specific = model.count_parameters()
        if isinstance(model_specific, dict):
            metrics.update(model_specific)
        elif isinstance(model_specific, tuple):
            orig, comp = model_specific
            metrics['original_parameters'] = orig
            metrics['compressed_parameters'] = comp
        # else: single int value, already counted in total_parameters
    
    if hasattr(model, 'get_all_time_constants'):
        tau_dict = model.get_all_time_constants()
        metrics['time_constants'] = tau_dict
    
    if hasattr(model, 'get_eigenmode_analysis'):
        eigenmodes = model.get_eigenmode_analysis()
        metrics['eigenmodes'] = eigenmodes
    
    return metrics


def benchmark_inference_speed(model: nn.Module,
                             V: torch.Tensor,
                             t: torch.Tensor,
                             num_runs: int = 100,
                             warmup_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark inference speed
    
    Args:
        model: Model to benchmark
        V: Voltage input
        t: Time input
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        
    Returns:
        speed_metrics: Inference speed statistics
    """
    model.eval()
    device = next(model.parameters()).device
    
    V = V.to(device)
    t = t.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            if hasattr(model, 'forward') and 'states' in model.forward.__code__.co_varnames:
                _ = model(V, t)
            else:
                _ = model(V, t)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            
            if hasattr(model, 'forward') and 'states' in model.forward.__code__.co_varnames:
                _ = model(V, t)
            else:
                _ = model(V, t)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.perf_counter()
            
            times.append(end - start)
    
    times = np.array(times)
    
    metrics = {
        'mean_time_ms': float(np.mean(times) * 1000),
        'std_time_ms': float(np.std(times) * 1000),
        'min_time_ms': float(np.min(times) * 1000),
        'max_time_ms': float(np.max(times) * 1000),
        'throughput_samples_per_sec': float(len(V) / np.mean(times))
    }
    
    return metrics


def compute_all_metrics(models_dict: Dict[str, nn.Module],
                       dataset: dict,
                       dt: float,
                       output_dir: str = './results') -> Dict[str, Dict]:
    """
    Compute all metrics for all models (Chapter 4 results)
    
    Args:
        models_dict: Dictionary of models {'name': model}
        dataset: Test dataset
        dt: Time step
        output_dir: Directory to save plots
        
    Returns:
        all_metrics: Complete metrics dictionary
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Computing Chapter 4 Experimental Results")
    print(f"{'='*60}")
    
    all_metrics = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    V_test = dataset['test']['V'].to(device)
    t_test = dataset['test']['t'].to(device)
    I_test = dataset['test']['I'].cpu().numpy()
    
    # Get baseline parameter count
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
        
        # 4. Inference speed
        speed_metrics = benchmark_inference_speed(model, V_test[:256], t_test[:256])
        metrics['speed'] = speed_metrics
        
        all_metrics[model_name] = metrics
        
        # Print summary
        print(f"  Time-domain MSE: {mse:.6e}")
        print(f"  Spectral MAE: {spectral_metrics['spectral_mae']:.6e}")
        print(f"  High-freq error: {spectral_metrics['high_freq_error']:.6e}")
        print(f"  Parameters: {compression_metrics['total_parameters']:,}")
        if 'reduction_percent' in compression_metrics:
            print(f"  Compression: {compression_metrics['reduction_percent']:.1f}% reduction")
        print(f"  Inference time: {speed_metrics['mean_time_ms']:.3f} ± {speed_metrics['std_time_ms']:.3f} ms")
    
    # Comparative analysis
    print(f"\n{'='*60}")
    print("Comparative Summary")
    print(f"{'='*60}")
    
    # Create comparison table
    print(f"\n{'Model':<20} {'Params':>12} {'Reduction':>12} {'MSE':>12} {'HF-Error':>12} {'Speed (ms)':>12}")
    print("-" * 90)
    
    for model_name, metrics in all_metrics.items():
        params = metrics['compression']['total_parameters']
        reduction = metrics['compression'].get('reduction_percent', 0.0)
        mse = metrics['time_domain']['mse']
        hf_error = metrics['spectral']['high_freq_error']
        speed = metrics['speed']['mean_time_ms']
        
        print(f"{model_name:<20} {params:>12,} {reduction:>11.1f}% {mse:>12.2e} {hf_error:>12.2e} {speed:>12.3f}")
    
    # Save metrics to file
    import json
    metrics_file = os.path.join(output_dir, 'chapter4_metrics.json')
    with open(metrics_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_safe_metrics = {}
        for model_name, model_metrics in all_metrics.items():
            json_safe_metrics[model_name] = {}
            for key, value in model_metrics.items():
                if isinstance(value, dict):
                    json_safe_metrics[model_name][key] = {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in value.items()
                        if not isinstance(v, dict)  # Skip nested dicts with arrays
                    }
                else:
                    json_safe_metrics[model_name][key] = value
        
        json.dump(json_safe_metrics, f, indent=2)
    
    print(f"\nMetrics saved to {metrics_file}")
    
    return all_metrics


if __name__ == "__main__":
    print("Testing evaluation metrics...")
    
    # Generate test data
    dt = 1e-7
    t = np.arange(0, 1e-3, dt)
    I_true = 0.5 * np.sin(2 * np.pi * 1e3 * t) + 0.1 * np.sin(2 * np.pi * 30e3 * t)
    I_pred = I_true + 0.05 * np.random.randn(len(I_true))
    
    metrics = compute_spectral_accuracy(I_pred, I_true, dt, 'test_spectral.png')
    print("Spectral metrics:", metrics)