"""
xLSTM-PINN Teacher Network
Implements Extended LSTM with Physics-Informed loss for high-frequency transient modeling
"""

import torch
import torch.nn as nn
import sys
import os

# Add xlstm-main to path for imports
xlstm_path = os.path.join(os.path.dirname(__file__), '..', '..', 'xlstm-main')
if xlstm_path not in sys.path:
    sys.path.insert(0, xlstm_path)

from typing import Tuple, Optional


class SimplifiedSLSTMCell(nn.Module):
    """
    Simplified sLSTM cell for time-series (adapted from xlstm-main)
    Uses exponential gating without CUDA dependencies
    """
    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Input projection to gates (i, f, c, o)
        self.W_gates = nn.Linear(input_size, 4 * hidden_size)
        
        # Recurrent connections (zeros initialization as per xLSTM paper)
        self.R_gates = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        nn.init.zeros_(self.R_gates.weight)
        
        # Forget gate bias initialization (power-law as per xlstm)
        with torch.no_grad():
            self.W_gates.bias[hidden_size:2*hidden_size] = torch.linspace(3.0, 6.0, hidden_size)
    
    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through sLSTM cell
        
        Args:
            x: Input tensor [batch, input_size]
            state: Tuple of (h, c) or None
            
        Returns:
            h: Hidden state [batch, hidden_size]
            (h, c): New state tuple
        """
        batch_size = x.size(0)
        
        if state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h, c = state
        
        # Compute gates
        gates = self.W_gates(x) + self.R_gates(h)
        
        # Split into individual gates
        i, f, g, o = gates.chunk(4, dim=1)
        
        # Apply activations (exponential for i, sigmoid/exp for f as per xLSTM)
        i = torch.exp(i)  # Exponential input gate (key xLSTM innovation)
        f = torch.sigmoid(f)  # Forget gate (can also use exp, but sigmoid is stable)
        g = torch.tanh(g)     # Cell gate
        o = torch.sigmoid(o)  # Output gate
        
        # Update cell state
        c_new = f * c + i * g
        
        # Compute hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, (h_new, c_new)


class SimplifiedMLSTMCell(nn.Module):
    """
    Simplified mLSTM cell with matrix memory
    Adapted from xlstm-main/xlstm/blocks/mlstm/cell.py
    """
    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        
        # Input and forget gates (computed from q, k, v)
        self.igate = nn.Linear(input_size, num_heads)
        self.fgate = nn.Linear(input_size, num_heads)
        
        # Output normalization
        self.out_norm = nn.LayerNorm(hidden_size)
        
        # Initialize forget gate bias (as per xlstm)
        with torch.no_grad():
            nn.init.zeros_(self.fgate.weight)
            self.fgate.bias.data = torch.linspace(3.0, 6.0, num_heads)
            nn.init.zeros_(self.igate.weight)
            nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)
    
    def forward(self, x: torch.Tensor, 
                state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass through mLSTM cell
        
        Args:
            x: Input [batch, input_size]
            state: Tuple of (C, n, m) matrix memory states or None
            
        Returns:
            h: Output [batch, hidden_size]
            new_state: Updated (C, n, m) tuple
        """
        batch_size = x.size(0)
        
        # Project to q, k, v
        q = self.W_q(x).view(batch_size, self.num_heads, self.head_dim)  # [B, NH, DH]
        k = self.W_k(x).view(batch_size, self.num_heads, self.head_dim)
        v = self.W_v(x).view(batch_size, self.num_heads, self.head_dim)
        
        # Compute gates - need proper shape for broadcasting
        i = torch.exp(self.igate(x)).unsqueeze(-1).unsqueeze(-1)  # [B, NH, 1, 1]
        f = torch.sigmoid(self.fgate(x)).unsqueeze(-1).unsqueeze(-1)  # [B, NH, 1, 1]
        
        # Initialize state if needed
        if state is None:
            C = torch.zeros(batch_size, self.num_heads, self.head_dim, self.head_dim,
                          device=x.device, dtype=x.dtype)
            n = torch.zeros(batch_size, self.num_heads, self.head_dim, 1,
                          device=x.device, dtype=x.dtype)
            m = torch.zeros(batch_size, self.num_heads, 1, 1,
                          device=x.device, dtype=x.dtype)
        else:
            C, n, m = state
        
        # Matrix memory update: C_t = f_t * C_{t-1} + i_t * (v_t @ k_t^T)
        # All shapes: C [B, NH, DH, DH], i/f [B, NH, 1, 1], v/k [B, NH, DH]
        C_new = f * C + i * (v.unsqueeze(-1) @ k.unsqueeze(-2))
        # For n: [B, NH, DH, 1], need f/i as [B, NH, 1, 1]
        n_new = f * n + i * v.unsqueeze(-1)
        # For m: [B, NH, 1, 1], f and i already have right shape
        m_new = f * m + i
        
        # Compute output: h = C_t @ q_t / (n_t^T @ q_t + m_t + eps)
        numerator = torch.matmul(C_new, q.unsqueeze(-1)).squeeze(-1)  # [B, NH, DH]
        # denominator: [B, NH, 1, DH] @ [B, NH, DH, 1] = [B, NH, 1, 1]
        denominator = torch.matmul(n_new.transpose(-2, -1), q.unsqueeze(-1)) + m_new + 1e-6
        
        h = numerator / denominator.squeeze(-1).squeeze(-1).unsqueeze(-1)  # Broadcast to [B, NH, DH]
        
        # Reshape and normalize
        h = h.reshape(batch_size, self.hidden_size)
        h = self.out_norm(h)
        
        return h, (C_new, n_new, m_new)


class xLSTMTeacher(nn.Module):
    """
    xLSTM-PINN Teacher network for high-frequency transient modeling
    Combines sLSTM and mLSTM blocks with physics-informed training
    """
    def __init__(self, 
                 input_dim: int = 2,  # [V, t]
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 output_dim: int = 1,  # Current I
                 use_mlstm: bool = True,
                 num_heads: int = 4):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.use_mlstm = use_mlstm
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, hidden_size)
        
        # xLSTM layers (alternating sLSTM and mLSTM as per paper)
        self.lstm_cells = nn.ModuleList()
        for i in range(num_layers):
            if use_mlstm and i % 2 == 1:
                self.lstm_cells.append(SimplifiedMLSTMCell(hidden_size, hidden_size, num_heads))
            else:
                self.lstm_cells.append(SimplifiedSLSTMCell(hidden_size, hidden_size, num_heads))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_dim)
        
    def forward(self, V: torch.Tensor, t: torch.Tensor, 
                states: Optional[list] = None) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through xLSTM-PINN
        
        Args:
            V: Voltage input [batch, 1]
            t: Time input [batch, 1]
            states: List of LSTM states or None
            
        Returns:
            I_pred: Predicted current [batch, 1]
            new_states: Updated LSTM states
        """
        # Concatenate inputs
        x = torch.cat([V, t], dim=1)
        
        # Input embedding
        h = self.input_proj(x)
        
        # Initialize states if needed
        if states is None:
            states = [None] * self.num_layers
        
        # Pass through xLSTM layers
        new_states = []
        for i, cell in enumerate(self.lstm_cells):
            h, state = cell(h, states[i])
            new_states.append(state)
        
        # Output projection
        I_pred = self.output_proj(h)
        
        return I_pred, new_states
    
    def compute_physics_loss(self, V: torch.Tensor, t: torch.Tensor, 
                           I_pred: torch.Tensor, I_true: torch.Tensor,
                           vteam_params: dict,
                           lambda_data: float = 1.0,
                           lambda_pde: float = 0.1,
                           lambda_ic: float = 0.1) -> Tuple[torch.Tensor, dict]:
        """
        Compute physics-informed loss
        
        Args:
            V: Voltage [batch, 1]
            t: Time [batch, 1] (requires_grad=True for AD)
            I_pred: Predicted current
            I_true: Ground truth current
            vteam_params: VTEAM model parameters
            lambda_data: Weight for data loss
            lambda_pde: Weight for PDE residual
            lambda_ic: Weight for initial condition
            
        Returns:
            total_loss: Combined loss
            losses: Dictionary of individual loss components
        """
        # Data loss (MSE)
        loss_data = torch.mean((I_pred - I_true) ** 2)
        
        # PDE loss: ∂I/∂t should match physics-based rate
        # For memristor: I = V / R(w), so ∂I/∂t involves ∂w/∂t
        # Simplified: check temporal consistency
        if t.requires_grad:
            dI_dt = torch.autograd.grad(
                I_pred.sum(), t, create_graph=True, retain_graph=True
            )[0]
            
            # Physical constraint: dI/dt should be bounded by RC time constant
            # For high-frequency: |dI/dt| < V_max * C / (R * dt)
            max_di_dt = 1e6  # Physical limit (A/s)
            loss_pde = torch.mean(torch.relu(torch.abs(dI_dt) - max_di_dt) ** 2)
        else:
            loss_pde = torch.tensor(0.0, device=I_pred.device)
        
        # Initial condition loss (enforce continuity)
        loss_ic = torch.mean((I_pred[0] - I_true[0]) ** 2)
        
        # Total loss
        total_loss = (lambda_data * loss_data + 
                     lambda_pde * loss_pde + 
                     lambda_ic * loss_ic)
        
        losses = {
            'total': total_loss.item(),
            'data': loss_data.item(),
            'pde': loss_pde.item() if isinstance(loss_pde, torch.Tensor) else 0.0,
            'ic': loss_ic.item()
        }
        
        return total_loss, losses


if __name__ == "__main__":
    # Test xLSTM Teacher
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = xLSTMTeacher(hidden_size=32, num_layers=2).to(device)
    
    # Test forward pass
    batch_size = 16
    V = torch.randn(batch_size, 1, device=device)
    t = torch.randn(batch_size, 1, device=device, requires_grad=True)
    
    I_pred, states = model(V, t)
    print(f"xLSTM Teacher test:")
    print(f"  Input: V{V.shape}, t{t.shape}")
    print(f"  Output: I_pred{I_pred.shape}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test physics loss
    I_true = torch.randn_like(I_pred)
    loss, loss_dict = model.compute_physics_loss(V, t, I_pred, I_true, {})
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Loss components: {loss_dict}")