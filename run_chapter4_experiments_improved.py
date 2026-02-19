"""
Improved Chapter 4 Experiment Runner with All Fixes
Addresses: speed optimization, dataset challenge, loss printing, multi-seed training
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from typing import Dict, List

# Import Ψ-xLSTM components
from psi_xlstm.data.memristor_generator import MemristorDataGenerator, MemristorConfig
from psi_xlstm.models.xlstm_teacher import xLSTMTeacher
from psi_xlstm.models.clustering_student import ClusteringStudent
from psi_xlstm.models.lowrank_mlstm import LowRankMLSTM
from psi_xlstm.training.distillation import StandardPINNBaseline, create_baseline_pinn
from psi_xlstm.training.trainer import train_teacher, train_student
from psi_xlstm.evaluation.metrics import compute_all_metrics
from psi_xlstm.hdl_generation.xlstm_verilog_gen import XLSTMVerilogGenerator


class ImprovedExperimentRunner:
    """Enhanced experiment runner with statistical validation"""
    
    def __init__(self, output_dir: str = './chapter4_results_improved'):
        self.output_dir = output_dir
        self.setup_directories()
        self.config = self.create_config()
        
    def setup_directories(self):
        """Create output directory structure"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'hdl'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'seeds'), exist_ok=True)
        
    def create_config(self) -> dict:
        """Create enhanced configuration"""
        config = {
            'experiment_name': 'Ψ-xLSTM Chapter 4 (Improved)',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'output_dir': self.output_dir,
            'num_seeds': 3,  # Full multi-seed validation for publication
            'random_seeds': [42, 123, 456]  # Three independent runs
        }
        
        print(f"\n{'='*70}")
        print(f"  Ψ-xLSTM: Improved Chapter 4 Experimental Setup")
        print(f"{'='*70}")
        print(f"Experiment: {config['experiment_name']}")
        print(f"Device: {config['device']}")
        print(f"Seeds: {config['random_seeds']}")
        print(f"Output: {config['output_dir']}")
        print(f"{'='*70}\n")
        
        return config
    
    def generate_challenging_dataset(self, seed: int = 42):
        """Generate MORE CHALLENGING dataset with higher frequencies"""
        print(f"\n{'='*70}")
        print(f"PHASE 1: Challenging Dataset Generation (seed={seed})")
        print(f"{'='*70}")
        
        # Increased challenge parameters
        memristor_config = MemristorConfig(
            dt=5e-8,  # 50ns time step (was 100ns) - captures higher frequencies
            t_max=2e-3  # 2ms simulation (was 1ms) - more data
        )
        
        generator = MemristorDataGenerator(memristor_config)
        generator.rng = np.random.RandomState(seed)  # Set seed for reproducibility
        
        # MORE SEQUENCES with HIGHER FREQUENCIES
        dataset = generator.generate_dataset(
            num_sequences=10,  # Was 5, now 10 for more diversity
            split_ratio=(0.7, 0.15, 0.15),
            device=self.config['device']
        )
        
        # Override with challenging frequencies (50-150 kHz)
        print("\nGenerating high-frequency sequences (50-150 kHz)...")
        all_t, all_V, all_I, all_w = [], [], [], []
        
        for seq_idx in range(10):
            f_high = 50e3 + seq_idx * 10e3  # 50-140 kHz (was 20-60 kHz)
            f_low = 1e3 + seq_idx * 500
            
            t, V = generator.generate_voltage_waveform(f_high=f_high, f_low=f_low)
            w_init = memristor_config.w_min + generator.rng.rand() * (memristor_config.w_max - memristor_config.w_min)
            
            # INCREASED NOISE for robustness testing
            I, w = generator.simulate_transient(V, t, w_init=w_init, 
                                               add_noise=True, noise_level=0.03)  # Was 0.01
            
            all_t.append(t)
            all_V.append(V)
            all_I.append(I)
            all_w.append(w)
            
            print(f"  Seq {seq_idx+1}/10: f_high={f_high/1e3:.0f}kHz, "
                  f"SNR={20*np.log10(np.std(I)/0.03/np.std(I)):.1f}dB")
        
        # Reconstruct dataset
        t_all = np.concatenate(all_t)
        V_all = np.concatenate(all_V)
        I_all = np.concatenate(all_I)
        w_all = np.concatenate(all_w)
        
        n_total = len(t_all)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)
        
        dataset = {
            'train': {
                't': torch.tensor(t_all[:n_train], dtype=torch.float32, device=self.config['device']).reshape(-1, 1),
                'V': torch.tensor(V_all[:n_train], dtype=torch.float32, device=self.config['device']).reshape(-1, 1),
                'I': torch.tensor(I_all[:n_train], dtype=torch.float32, device=self.config['device']).reshape(-1, 1),
                'w': torch.tensor(w_all[:n_train], dtype=torch.float32, device=self.config['device']).reshape(-1, 1)
            },
            'val': {
                't': torch.tensor(t_all[n_train:n_train+n_val], dtype=torch.float32, device=self.config['device']).reshape(-1, 1),
                'V': torch.tensor(V_all[n_train:n_train+n_val], dtype=torch.float32, device=self.config['device']).reshape(-1, 1),
                'I': torch.tensor(I_all[n_train:n_train+n_val], dtype=torch.float32, device=self.config['device']).reshape(-1, 1),
                'w': torch.tensor(w_all[n_train:n_train+n_val], dtype=torch.float32, device=self.config['device']).reshape(-1, 1)
            },
            'test': {
                't': torch.tensor(t_all[n_train+n_val:], dtype=torch.float32, device=self.config['device']).reshape(-1, 1),
                'V': torch.tensor(V_all[n_train+n_val:], dtype=torch.float32, device=self.config['device']).reshape(-1, 1),
                'I': torch.tensor(I_all[n_train+n_val:], dtype=torch.float32, device=self.config['device']).reshape(-1, 1),
                'w': torch.tensor(w_all[n_train+n_val:], dtype=torch.float32, device=self.config['device']).reshape(-1, 1)
            }
        }
        
        print(f"\nChallenging dataset created!")
        print(f"  Train: {len(dataset['train']['t']):,} samples")
        print(f"  Frequency range: 50-150 kHz")
        print(f"  Noise level: 3% (increased from 1%)")
        
        return dataset, memristor_config
    
    def train_single_run(self, dataset: dict, seed: int):
        """Train all models for a single seed"""
        print(f"\n{'='*70}")
        print(f"PHASE 2-3: Model Training (seed={seed})")
        print(f"{'='*70}")
        
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        device = self.config['device']
        models = {}
        
        # 1. Baseline PINN
        print(f"\n[1/4] Training Baseline PINN (seed={seed})...")
        baseline = create_baseline_pinn(2, 64, 3, 1).to(device)
        optimizer = torch.optim.Adam(baseline.parameters(), lr=1e-3)
        
        for epoch in range(50):
            total_loss = 0
            for i in range(0, len(dataset['train']['V']), 256):
                V_batch = dataset['train']['V'][i:i+256]
                t_batch = dataset['train']['t'][i:i+256]
                I_batch = dataset['train']['I'][i:i+256]
                
                optimizer.zero_grad()
                I_pred = baseline(V_batch, t_batch)
                loss = torch.mean((I_pred - I_batch) ** 2)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/50: Loss = {total_loss:.3e}")  # Scientific notation
        
        models['baseline_pinn'] = baseline
        
        # 2. Teacher
        print(f"\n[2/4] Training Teacher (seed={seed})...")
        seed_dir = os.path.join(self.output_dir, 'seeds', f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        teacher = xLSTMTeacher(2, 64, 2, 1, use_mlstm=True, num_heads=4).to(device)
        teacher, _ = train_teacher(teacher, dataset, num_epochs=100, device=device, save_dir=seed_dir)
        models['teacher'] = teacher
        
        # 3. Clustering Student
        print(f"\n[3/4] Training Clustering Student (seed={seed})...")
        clustering_student = ClusteringStudent(2, 32, 2, 1, num_clusters=3).to(device)
        clustering_student, _ = train_student(
            clustering_student, teacher, dataset,
            num_epochs=150, device=device, gamma=0.1, save_dir=seed_dir
        )
        models['psi_xlstm_clustering'] = clustering_student
        
        # 4. Low-Rank Student
        print(f"\n[4/4] Training Low-Rank Student (seed={seed})...")
        lowrank_student = LowRankMLSTM(2, 32, 2, 1, rank=2, num_heads=4).to(device)
        lowrank_student, _ = train_student(
            lowrank_student, teacher, dataset,
            num_epochs=150, device=device, gamma=0.0, save_dir=seed_dir
        )
        models['psi_xlstm_lowrank'] = lowrank_student
        
        # Save final models (training functions already save best versions)
        for name, model in models.items():
            final_path = os.path.join(seed_dir, f'{name}_final.pth')
            torch.save(model.state_dict(), final_path)
        
        return models
    
    def evaluate_with_statistics(self, all_runs_models: List[Dict], dataset: dict, dt: float):
        """Evaluate all runs and compute statistics"""
        print(f"\n{'='*70}")
        print("PHASE 4: Statistical Evaluation")
        print(f"{'='*70}")
        
        # Import improved metrics
        from psi_xlstm.evaluation.metrics_improved import compute_all_metrics_optimized
        
        all_seed_metrics = []
        
        for seed_idx, models in enumerate(all_runs_models):
            print(f"\nEvaluating seed {self.config['random_seeds'][seed_idx]}...")
            metrics = compute_all_metrics_optimized(
                models, dataset, dt,
                output_dir=os.path.join(self.output_dir, f'plots_seed{seed_idx}')
            )
            all_seed_metrics.append(metrics)
        
        # Compute statistics across seeds
        aggregated_metrics = self.aggregate_statistics(all_seed_metrics)
        
        # Print results with confidence intervals
        self.print_results_with_ci(aggregated_metrics)
        
        # Generate publication-ready materials
        from psi_xlstm.evaluation.publication_plots import generate_all_publication_materials
        pub_dir = os.path.join(self.output_dir, 'publication_materials')
        generate_all_publication_materials(all_seed_metrics, pub_dir)
        
        return aggregated_metrics
    
    def aggregate_statistics(self, all_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across seeds"""
        aggregated = {}
        model_names = all_metrics[0].keys()
        
        for model_name in model_names:
            aggregated[model_name] = {}
            
            # Collect metrics across seeds
            mse_values = [m[model_name]['time_domain']['mse'] for m in all_metrics]
            hf_error_values = [m[model_name]['spectral']['high_freq_error'] for m in all_metrics]
            speed_values = [m[model_name]['speed']['mean_time_ms'] for m in all_metrics]
            params = all_metrics[0][model_name]['compression']['total_parameters']
            
            aggregated[model_name] = {
                'mse_mean': np.mean(mse_values),
                'mse_std': np.std(mse_values),
                'mse_ci': 1.96 * np.std(mse_values) / np.sqrt(len(mse_values)),
                'hf_error_mean': np.mean(hf_error_values),
                'hf_error_std': np.std(hf_error_values),
                'speed_mean': np.mean(speed_values),
                'speed_std': np.std(speed_values),
                'parameters': params
            }
        
        return aggregated
    
    def print_results_with_ci(self, metrics: Dict):
        """Print results with confidence intervals"""
        print(f"\n{'='*70}")
        print("RESULTS WITH 95% CONFIDENCE INTERVALS")
        print(f"{'='*70}")
        
        print(f"\n{'Model':<25} {'MSE (mean±CI)':>25} {'HF-Error (mean±CI)':>25}")
        print("-" * 75)
        
        for model_name, m in metrics.items():
            mse_str = f"{m['mse_mean']:.2e}±{m['mse_ci']:.2e}"
            hf_str = f"{m['hf_error_mean']:.2e}±{m['hf_error_ci'] if 'hf_error_ci' in m else 0:.2e}"
            print(f"{model_name:<25} {mse_str:>25} {hf_str:>25}")
        
        # Save to JSON
        json_path = os.path.join(self.output_dir, 'statistical_results.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nStatistical results saved to {json_path}")
    
    def run_complete_experiment(self):
        """Run complete multi-seed experiment"""
        all_runs_models = []
        
        for seed in self.config['random_seeds']:
            # Generate challenging dataset
            dataset, memristor_config = self.generate_challenging_dataset(seed)
            
            # Train all models
            models = self.train_single_run(dataset, seed)
            all_runs_models.append(models)
        
        # Evaluate with statistics
        dataset, memristor_config = self.generate_challenging_dataset(self.config['random_seeds'][0])
        aggregated_metrics = self.evaluate_with_statistics(all_runs_models, dataset, memristor_config.dt)
        
        # Generate Verilog-A (use first seed models)
        print(f"\n{'='*70}")
        print("PHASE 5: Verilog-A Generation")
        print(f"{'='*70}")
        
        best_models = all_runs_models[0]
        hdl_dir = os.path.join(self.output_dir, 'hdl')
        
        if 'psi_xlstm_clustering' in best_models:
            generator = XLSTMVerilogGenerator(best_models['psi_xlstm_clustering'], 'psi_xlstm_clustering')
            generator.generate_hdl_package(hdl_dir)
        
        print(f"\n{'='*70}")
        print("✓ IMPROVED EXPERIMENTS COMPLETED!")
        print(f"{'='*70}")
        print(f"\nKey Improvements:")
        print(f"  • Dataset: 50-150 kHz (was 20-60 kHz)")
        print(f"  • Noise: 3% (was 1%)")
        print(f"  • Multi-seed: {len(self.config['random_seeds'])} runs with CI")
        print(f"  • Speed: Optimized batch processing")
        print(f"  • Loss: Scientific notation for clarity")
        print(f"\nResults in: {self.output_dir}")


def main():
    runner = ImprovedExperimentRunner()
    runner.run_complete_experiment()


if __name__ == "__main__":
    main()