"""
Memristor Data Generator for High-Frequency Transient Modeling
Implements VTEAM model with configurable high-frequency voltage waveforms
"""

import numpy as np
import torch
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class MemristorConfig:
    """Configuration for VTEAM memristor model"""
    # VTEAM model parameters
    v_on: float = 0.5      # Threshold voltage for ON switching (V)
    v_off: float = -0.5    # Threshold voltage for OFF switching (V)
    k_on: float = 8000.0   # ON switching rate (nm/s)
    k_off: float = -8000.0 # OFF switching rate (nm/s)
    alpha_on: float = 3.0  # ON nonlinearity
    alpha_off: float = 3.0 # OFF nonlinearity
    
    # Device parameters
    w_min: float = 1.0     # Minimum state variable (nm)
    w_max: float = 10.0    # Maximum state variable (nm)
    R_on: float = 100.0    # ON resistance (Ohms)
    R_off: float = 10000.0 # OFF resistance (Ohms)
    
    # Simulation parameters
    dt: float = 1e-7       # Time step (100 ns for high-frequency)
    t_max: float = 1e-3    # Total simulation time (1 ms)
    
    def __post_init__(self):
        self.num_steps = int(self.t_max / self.dt)


class MemristorDataGenerator:
    """Generate realistic memristor transient data with high-frequency dynamics"""
    
    def __init__(self, config: MemristorConfig = None):
        self.config = config or MemristorConfig()
        self.rng = np.random.RandomState(42)
        
    def generate_voltage_waveform(self, 
                                   f_high: float = 50e3,  # 50 kHz
                                   f_low: float = 1e3,    # 1 kHz
                                   add_transients: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complex voltage waveform with high-frequency components
        
        Args:
            f_high: High-frequency component (Hz)
            f_low: Low-frequency component (Hz)
            add_transients: Add sharp voltage steps for transient testing
            
        Returns:
            t: Time array
            V: Voltage array
        """
        t = np.linspace(0, self.config.t_max, self.config.num_steps)
        
        # Multi-frequency sinusoidal
        V = 0.5 * np.sin(2 * np.pi * f_low * t)      # Low-frequency carrier
        V += 0.3 * np.sin(2 * np.pi * f_high * t)    # High-frequency component
        
        if add_transients:
            # Add sharp voltage steps to test spectral bias
            step_times = [0.2e-3, 0.5e-3, 0.8e-3]
            for t_step in step_times:
                step_idx = int(t_step / self.config.dt)
                if step_idx < len(V):
                    # Sharp step with exponential relaxation
                    step_mask = t > t_step
                    V[step_mask] += 0.4 * np.exp(-1e4 * (t[step_mask] - t_step))
        
        return t, V
    
    def _vteam_switching_function(self, V: float, w: float) -> float:
        """
        VTEAM switching rate: dw/dt = f(V, w)
        
        From memristor physics:
        dw/dt = k_on * (w_max - w) * ((V/v_on - 1)^α_on)  if V > v_on
        dw/dt = k_off * w * ((V/v_off - 1)^α_off)         if V < v_off
        dw/dt = 0                                          otherwise
        """
        cfg = self.config
        
        if V > cfg.v_on:
            # ON switching
            f_on = cfg.k_on * (cfg.w_max - w) * np.power(V / cfg.v_on - 1, cfg.alpha_on)
            return max(0, f_on)  # Ensure non-negative
        elif V < cfg.v_off:
            # OFF switching
            f_off = cfg.k_off * w * np.power(V / cfg.v_off - 1, cfg.alpha_off)
            return min(0, f_off)  # Ensure non-positive
        else:
            return 0.0
    
    def _compute_current(self, V: float, w: float) -> float:
        """
        Compute current through memristor
        I = V / R(w), where R(w) is state-dependent resistance
        """
        cfg = self.config
        # Linear interpolation between R_on and R_off
        w_normalized = (w - cfg.w_min) / (cfg.w_max - cfg.w_min)
        w_normalized = np.clip(w_normalized, 0, 1)
        
        R = cfg.R_on * w_normalized + cfg.R_off * (1 - w_normalized)
        return V / R
    
    def simulate_transient(self, V: np.ndarray, t: np.ndarray, 
                           w_init: float = 5.0,
                           add_noise: bool = True,
                           noise_level: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate memristor transient response using explicit Euler integration
        
        Args:
            V: Voltage waveform
            t: Time array
            w_init: Initial state variable value
            add_noise: Add measurement noise
            noise_level: Relative noise level (fraction of signal)
            
        Returns:
            I: Current array
            w: State variable array
        """
        cfg = self.config
        num_steps = len(t)
        
        # Initialize arrays
        w = np.zeros(num_steps)
        I = np.zeros(num_steps)
        
        w[0] = w_init
        I[0] = self._compute_current(V[0], w[0])
        
        # Explicit Euler integration
        for i in range(1, num_steps):
            dt = t[i] - t[i-1]
            
            # State update: w_new = w_old + dt * dw/dt
            dw_dt = self._vteam_switching_function(V[i-1], w[i-1])
            w[i] = w[i-1] + dt * dw_dt
            
            # Enforce bounds
            w[i] = np.clip(w[i], cfg.w_min, cfg.w_max)
            
            # Compute current
            I[i] = self._compute_current(V[i], w[i])
        
        # Add realistic measurement noise
        if add_noise:
            I_noise = I + noise_level * np.std(I) * self.rng.randn(num_steps)
            return I_noise, w
        
        return I, w
    
    def generate_dataset(self, 
                        num_sequences: int = 5,
                        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
                        ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate complete dataset for training/validation/testing
        
        Returns:
            dataset: Dictionary with 'train', 'val', 'test' splits
                    Each containing 't', 'V', 'I', 'w' tensors
        """
        print(f"Generating memristor dataset with {num_sequences} sequences...")
        
        all_t, all_V, all_I, all_w = [], [], [], []
        
        for seq_idx in range(num_sequences):
            # Generate diverse voltage waveforms
            f_high = 20e3 + seq_idx * 10e3  # Vary frequency: 20-60 kHz
            f_low = 500 + seq_idx * 500     # Vary low freq: 500-2500 Hz
            
            t, V = self.generate_voltage_waveform(f_high=f_high, f_low=f_low)
            
            # Random initial state
            w_init = self.config.w_min + self.rng.rand() * (self.config.w_max - self.config.w_min)
            
            I, w = self.simulate_transient(V, t, w_init=w_init)
            
            all_t.append(t)
            all_V.append(V)
            all_I.append(I)
            all_w.append(w)
            
            print(f"  Sequence {seq_idx+1}/{num_sequences}: "
                  f"f_high={f_high/1e3:.1f}kHz, I_range=[{I.min():.2e}, {I.max():.2e}]")
        
        # Concatenate all sequences
        t_all = np.concatenate(all_t)
        V_all = np.concatenate(all_V)
        I_all = np.concatenate(all_I)
        w_all = np.concatenate(all_w)
        
        # Create train/val/test splits
        n_total = len(t_all)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])
        
        # Convert to torch tensors
        dataset = {
            'train': {
                't': torch.tensor(t_all[:n_train], dtype=torch.float32, device=device).reshape(-1, 1),
                'V': torch.tensor(V_all[:n_train], dtype=torch.float32, device=device).reshape(-1, 1),
                'I': torch.tensor(I_all[:n_train], dtype=torch.float32, device=device).reshape(-1, 1),
                'w': torch.tensor(w_all[:n_train], dtype=torch.float32, device=device).reshape(-1, 1)
            },
            'val': {
                't': torch.tensor(t_all[n_train:n_train+n_val], dtype=torch.float32, device=device).reshape(-1, 1),
                'V': torch.tensor(V_all[n_train:n_train+n_val], dtype=torch.float32, device=device).reshape(-1, 1),
                'I': torch.tensor(I_all[n_train:n_train+n_val], dtype=torch.float32, device=device).reshape(-1, 1),
                'w': torch.tensor(w_all[n_train:n_train+n_val], dtype=torch.float32, device=device).reshape(-1, 1)
            },
            'test': {
                't': torch.tensor(t_all[n_train+n_val:], dtype=torch.float32, device=device).reshape(-1, 1),
                'V': torch.tensor(V_all[n_train+n_val:], dtype=torch.float32, device=device).reshape(-1, 1),
                'I': torch.tensor(I_all[n_train+n_val:], dtype=torch.float32, device=device).reshape(-1, 1),
                'w': torch.tensor(w_all[n_train+n_val:], dtype=torch.float32, device=device).reshape(-1, 1)
            }
        }
        
        print(f"\nDataset splits:")
        print(f"  Train: {len(dataset['train']['t'])} samples")
        print(f"  Val:   {len(dataset['val']['t'])} samples")
        print(f"  Test:  {len(dataset['test']['t'])} samples")
        
        return dataset
    
    def get_physics_function(self):
        """Return physics function for PDE loss computation"""
        def physics_residual(I_pred, w_pred, V, t):
            """
            Physics residual for memristor:
            dw/dt = f(V, w)
            I = V / R(w)
            """
            # This will be used in PINN loss computation
            return self._vteam_switching_function, self._compute_current
        
        return physics_residual


if __name__ == "__main__":
    # Test data generation
    generator = MemristorDataGenerator()
    dataset = generator.generate_dataset(num_sequences=3)
    
    print("\nDataset statistics:")
    for split in ['train', 'val', 'test']:
        I = dataset[split]['I'].cpu().numpy()
        print(f"{split.upper()}: I_mean={I.mean():.2e}, I_std={I.std():.2e}")