"""
Generate publication-ready plots and tables for manuscript
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from typing import Dict, List
import os

# Set publication-quality matplotlib settings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.dpi'] = 300


def generate_comparison_bar_plot(all_seed_metrics: List[Dict], output_path: str):
    """
    Generate bar plot comparing models across metrics with error bars
    Publication-ready figure for manuscript
    """
    model_names = list(all_seed_metrics[0].keys())
    metrics_to_plot = ['time_domain', 'spectral', 'compression', 'speed']
    
    # Aggregate data across seeds
    aggregated = {}
    for model_name in model_names:
        mse_values = [m[model_name]['time_domain']['mse'] for m in all_seed_metrics]
        hf_error_values = [m[model_name]['spectral']['high_freq_error'] for m in all_seed_metrics]
        params = all_seed_metrics[0][model_name]['compression']['total_parameters']
        latency_values = [m[model_name]['speed']['latency_per_sample_ms'] * 1000 for m in all_seed_metrics]  # Convert to μs
        
        aggregated[model_name] = {
            'mse_mean': np.mean(mse_values),
            'mse_std': np.std(mse_values),
            'hf_error_mean': np.mean(hf_error_values),
            'hf_error_std': np.std(hf_error_values),
            'params': params,
            'latency_mean': np.mean(latency_values),
            'latency_std': np.std(latency_values)
        }
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Ψ-xLSTM: Model Comparison Across Metrics', fontsize=14, fontweight='bold')
    
    x = np.arange(len(model_names))
    width = 0.6
    
    # Clean model names for display
    display_names = {
        'baseline_pinn': 'Baseline\nPINN',
        'teacher': 'xLSTM-PINN\nTeacher',
        'psi_xlstm_clustering': 'Ψ-xLSTM\nClustering',
        'psi_xlstm_lowrank': 'Ψ-xLSTM\nLow-Rank'
    }
    
    # 1. Time-domain MSE
    ax = axes[0, 0]
    mse_means = [aggregated[m]['mse_mean'] for m in model_names]
    mse_stds = [aggregated[m]['mse_std'] for m in model_names]
    bars = ax.bar(x, mse_means, width, yerr=mse_stds, capsize=5, 
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax.set_ylabel('MSE')
    ax.set_title('(a) Time-Domain Accuracy', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([display_names[m] for m in model_names], rotation=0)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. High-frequency error
    ax = axes[0, 1]
    hf_means = [aggregated[m]['hf_error_mean'] for m in model_names]
    hf_stds = [aggregated[m]['hf_error_std'] for m in model_names]
    ax.bar(x, hf_means, width, yerr=hf_stds, capsize=5,
           color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax.set_ylabel('High-Freq Error')
    ax.set_title('(b) High-Frequency (>20kHz) Performance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([display_names[m] for m in model_names], rotation=0)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Model size
    ax = axes[1, 0]
    params = [aggregated[m]['params'] for m in model_names]
    ax.bar(x, params, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax.set_ylabel('Parameters')
    ax.set_title('(c) Model Compression', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([display_names[m] for m in model_names], rotation=0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add compression percentages
    baseline_params = aggregated['teacher']['params']
    for i, (name, param_count) in enumerate(zip(model_names, params)):
        reduction = (1 - param_count / baseline_params) * 100
        if reduction > 0:
            ax.text(i, param_count, f'-{reduction:.0f}%', ha='center', va='bottom', fontsize=8)
    
    # 4. Inference speed
    ax = axes[1, 1]
    latency_means = [aggregated[m]['latency_mean'] for m in model_names]
    latency_stds = [aggregated[m]['latency_std'] for m in model_names]
    ax.bar(x, latency_means, width, yerr=latency_stds, capsize=5,
           color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.8)
    ax.set_ylabel('Latency per Sample (μs)')
    ax.set_title('(d) Inference Speed', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([display_names[m] for m in model_names], rotation=0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add speedup annotations
    baseline_latency = aggregated['teacher']['latency_mean']
    for i, (name, lat) in enumerate(zip(model_names, latency_means)):
        speedup = baseline_latency / lat
        if speedup > 1:
            ax.text(i, lat, f'{speedup:.1f}x', ha='center', va='bottom', fontsize=8, color='green')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Publication bar plot saved to {output_path}")


def generate_compression_accuracy_tradeoff(all_seed_metrics: List[Dict], output_path: str):
    """
    Generate Pareto frontier plot: Compression vs Accuracy
    """
    model_names = list(all_seed_metrics[0].keys())
    
    # Aggregate data
    data = {}
    for model_name in model_names:
        mse_values = [m[model_name]['time_domain']['mse'] for m in all_seed_metrics]
        params = all_seed_metrics[0][model_name]['compression']['total_parameters']
        teacher_params = all_seed_metrics[0]['teacher']['compression']['total_parameters']
        compression_ratio = (1 - params / teacher_params) * 100
        
        data[model_name] = {
            'compression': compression_ratio,
            'mse_mean': np.mean(mse_values),
            'mse_std': np.std(mse_values)
        }
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = {'baseline_pinn': '#1f77b4', 'teacher': '#ff7f0e', 
              'psi_xlstm_clustering': '#2ca02c', 'psi_xlstm_lowrank': '#d62728'}
    labels = {'baseline_pinn': 'Baseline PINN', 'teacher': 'xLSTM Teacher',
              'psi_xlstm_clustering': 'Ψ-xLSTM Clustering', 'psi_xlstm_lowrank': 'Ψ-xLSTM Low-Rank'}
    
    for model_name in model_names:
        d = data[model_name]
        ax.errorbar(d['compression'], d['mse_mean'], yerr=d['mse_std'],
                   fmt='o', markersize=10, capsize=5, label=labels[model_name],
                   color=colors[model_name], linewidth=2)
    
    ax.set_xlabel('Compression Ratio (%)', fontweight='bold')
    ax.set_ylabel('Test MSE', fontweight='bold')
    ax.set_title('Compression-Accuracy Tradeoff', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Tradeoff plot saved to {output_path}")


def generate_latex_tables(all_seed_metrics: List[Dict], output_dir: str):
    """
    Generate LaTeX tables ready for manuscript
    """
    model_names = list(all_seed_metrics[0].keys())
    
    # Table 1: Main Results
    table1 = []
    table1.append("\\begin{table}[htbp]")
    table1.append("\\centering")
    table1.append("\\caption{Performance comparison of Ψ-xLSTM models on high-frequency memristor dynamics (50-150 kHz, 3\\% noise). Results averaged over 3 random seeds.}")
    table1.append("\\label{tab:main_results}")
    table1.append("\\begin{tabular}{lcccc}")
    table1.append("\\hline")
    table1.append("\\textbf{Model} & \\textbf{Parameters} & \\textbf{MSE ($\\times 10^{-8}$)} & \\textbf{HF-Error ($\\times 10^{-7}$)} & \\textbf{Latency (μs)} \\\\")
    table1.append("\\hline")
    
    labels = {'baseline_pinn': 'Baseline PINN', 'teacher': 'xLSTM-PINN Teacher',
              'psi_xlstm_clustering': 'Ψ-xLSTM Clustering', 'psi_xlstm_lowrank': 'Ψ-xLSTM Low-Rank'}
    
    for model_name in model_names:
        mse_values = [m[model_name]['time_domain']['mse'] for m in all_seed_metrics]
        hf_values = [m[model_name]['spectral']['high_freq_error'] for m in all_seed_metrics]
        latency_values = [m[model_name]['speed']['latency_per_sample_ms'] * 1000 for m in all_seed_metrics]
        params = all_seed_metrics[0][model_name]['compression']['total_parameters']
        
        mse_mean = np.mean(mse_values) * 1e8
        mse_std = np.std(mse_values) * 1e8
        hf_mean = np.mean(hf_values) * 1e7
        hf_std = np.std(hf_values) * 1e7
        lat_mean = np.mean(latency_values)
        lat_std = np.std(latency_values)
        
        row = f"{labels[model_name]} & {params:,} & ${mse_mean:.2f} \\pm {mse_std:.2f}$ & ${hf_mean:.2f} \\pm {hf_std:.2f}$ & ${lat_mean:.3f} \\pm {lat_std:.3f}$ \\\\"
        table1.append(row)
    
    table1.append("\\hline")
    table1.append("\\end{tabular}")
    table1.append("\\end{table}")
    
    # Save table 1
    with open(os.path.join(output_dir, 'table1_main_results.tex'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(table1))
    
    print(f"  LaTeX tables saved to {output_dir}")


def generate_csv_tables(all_seed_metrics: List[Dict], output_dir: str):
    """
    Generate CSV tables with all metrics for each seed
    """
    model_names = list(all_seed_metrics[0].keys())
    
    # Individual seed results
    for seed_idx, metrics in enumerate(all_seed_metrics):
        rows = []
        for model_name in model_names:
            m = metrics[model_name]
            row = {
                'Model': model_name,
                'Seed': seed_idx,
                'Parameters': m['compression']['total_parameters'],
                'MSE': m['time_domain']['mse'],
                'MAE': m['time_domain']['mae'],
                'RMSE': m['time_domain']['rmse'],
                'Spectral_MAE': m['spectral']['spectral_mae'],
                'HF_Error': m['spectral']['high_freq_error'],
                'LF_Error': m['spectral']['low_freq_error'],
                'Freq_Correlation': m['spectral']['freq_correlation'],
                'Latency_ms': m['speed']['latency_per_sample_ms'],
                'Throughput_samples_per_sec': m['speed']['throughput_samples_per_sec']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, f'seed_{seed_idx}_results.csv'), index=False)
    
    # Aggregated results
    agg_rows = []
    for model_name in model_names:
        mse_values = [m[model_name]['time_domain']['mse'] for m in all_seed_metrics]
        hf_values = [m[model_name]['spectral']['high_freq_error'] for m in all_seed_metrics]
        latency_values = [m[model_name]['speed']['latency_per_sample_ms'] for m in all_seed_metrics]
        
        agg_row = {
            'Model': model_name,
            'Parameters': all_seed_metrics[0][model_name]['compression']['total_parameters'],
            'MSE_mean': np.mean(mse_values),
            'MSE_std': np.std(mse_values),
            'MSE_ci': 1.96 * np.std(mse_values) / np.sqrt(len(mse_values)),
            'HF_Error_mean': np.mean(hf_values),
            'HF_Error_std': np.std(hf_values),
            'HF_Error_ci': 1.96 * np.std(hf_values) / np.sqrt(len(hf_values)),
            'Latency_mean_us': np.mean(latency_values) * 1000,
            'Latency_std_us': np.std(latency_values) * 1000,
            'Latency_ci_us': 1.96 * np.std(latency_values) * 1000 / np.sqrt(len(latency_values))
        }
        agg_rows.append(agg_row)
    
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(os.path.join(output_dir, 'aggregated_results.csv'), index=False)
    
    print(f"  CSV tables saved to {output_dir}")


def generate_all_publication_materials(all_seed_metrics: List[Dict], output_dir: str):
    """
    Generate all publication-ready materials
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("Generating Publication-Ready Materials")
    print(f"{'='*70}")
    
    # Bar plots
    generate_comparison_bar_plot(all_seed_metrics, 
                                 os.path.join(output_dir, 'fig_comparison_bars.png'))
    
    # Tradeoff plot
    generate_compression_accuracy_tradeoff(all_seed_metrics,
                                          os.path.join(output_dir, 'fig_tradeoff.png'))
    
    # LaTeX tables
    generate_latex_tables(all_seed_metrics, output_dir)
    
    # CSV tables
    generate_csv_tables(all_seed_metrics, output_dir)
    
    print(f"\n✓ All publication materials generated in: {output_dir}")
    print(f"  • Figures: fig_comparison_bars.png, fig_tradeoff.png")
    print(f"  • LaTeX: table1_main_results.tex")
    print(f"  • CSV: seed_*_results.csv, aggregated_results.csv")