"""
Low-Rank Matrix Memory mLSTM
Implements compressed matrix memory via low-rank factorization (equations 8-9)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LowRankMLSTMCell(nn.Module):
    """
    mLSTM cell with low-rank factorized matrix memory
    Instead of C_t ∈ R^{d×d}, uses U @ C_tilde @ V^T where C_tilde ∈ R^{r×r}
    """
    def __init__(self, input_size: int, hidden_size: int, rank: int = 2, num_heads: int = 4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(input_size, hidden_size)
        self.W_k = nn.Linear(input_size, hidden_size)
        self.W_v = nn.Linear(input_size, hidden_size)
        
        # Low-rank factorization matrices
        # U: [num_heads, head_dim, rank]
        # V: [num_heads, head_dim, rank]
        self.U = nn.Parameter(torch.randn(num_heads, self.head_dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(num_heads, self.head_dim, rank) * 0.01)
        
        # Gates (operating in compressed space)
        self.igate = nn.Linear(input_size, num_heads)
        self.fgate = nn.Linear(input_size, num_heads)
        
        # Output normalization
        self.out_norm = nn.LayerNorm(hidden_size)
        
        # Initialize gates
        with torch.no_grad():
            nn.init.zeros_(self.fgate.weight)
            self.fgate.bias.data = torch.linspace(3.0, 6.0, num_heads)
            nn.init.zeros_(self.igate.weight)
            nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)
    
    def forward(self, x: torch.Tensor,
                state: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with low-rank matrix memory
        
        Args:
            x: Input [batch, input_size]
            state: Compressed matrix C_tilde [batch, num_heads, rank, rank] or None
            
        Returns:
            h: Output [batch, hidden_size]
            C_tilde_new: Updated compressed state
        """
        batch_size = x.size(0)
        
        # Project to q, k, v
        q = self.W_q(x).view(batch_size, self.num_heads, self.head_dim)  # [B, NH, DH]
        k = self.W_k(x).view(batch_size, self.num_heads, self.head_dim)
        v = self.W_v(x).view(batch_size, self.num_heads, self.head_dim)
        
        # Compute gates
        i = torch.exp(self.igate(x)).unsqueeze(-1).unsqueeze(-1)  # [B, NH, 1, 1]
        f = torch.sigmoid(self.fgate(x)).unsqueeze(-1).unsqueeze(-1)  # [B, NH, 1, 1]
        
        # Initialize compressed state if needed
        if state is None:
            C_tilde = torch.zeros(batch_size, self.num_heads, self.rank, self.rank,
                                device=x.device, dtype=x.dtype)
        else:
            C_tilde = state
        
        # Project v and k to compressed space
        # v_compressed = U^T @ v: [NH, rank, DH] @ [B, NH, DH, 1] -> [B, NH, rank, 1]
        v_compressed = torch.einsum('ndr,bnd->bnr', self.U, v).unsqueeze(-1)  # [B, NH, r, 1]
        k_compressed = torch.einsum('ndr,bnd->bnr', self.V, k).unsqueeze(-1)  # [B, NH, r, 1]
        
        # Update compressed matrix memory (equation 9)
        # C_tilde_t = f_t * C_tilde_{t-1} + i_t * (v_comp @ k_comp^T)
        C_tilde_new = f * C_tilde + i * (v_compressed @ k_compressed.transpose(-2, -1))
        
        # Project query to compressed space
        q_compressed = torch.einsum('ndr,bnd->bnr', self.U, q)  # [B, NH, r]
        
        # Compute output in compressed space then reconstruct
        # h_compressed = C_tilde @ q_compressed
        h_compressed = torch.einsum('bnrr,bnr->bnr', C_tilde_new, q_compressed)
        
        # Reconstruct to full space: h = U @ h_compressed
        h = torch.einsum('ndr,bnr->bnd', self.U, h_compressed)  # [B, NH, DH]
        
        # Reshape and normalize
        h = h.reshape(batch_size, self.hidden_size)
        h = self.out_norm(h)
        
        return h, C_tilde_new
    
    def get_compression_ratio(self) -> float:
        """
        Calculate compression ratio compared to full matrix memory
        
        Full: d×d per head = head_dim^2
        Compressed: 2×d×r (U and V) + r^2 (C_tilde) ≈ 2×d×r for small r
        """
        full_size = self.num_heads * (self.head_dim ** 2)
        compressed_size = self.num_heads * (2 * self.head_dim * self.rank + self.rank ** 2)
        return compressed_size / full_size


class LowRankMLSTM(nn.Module):
    """
    Complete low-rank mLSTM network for Ψ-xLSTM student
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_size: int = 32,
                 num_layers: int = 2,
                 output_dim: int = 1,
                 rank: int = 2,
                 num_heads: int = 4):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.rank = rank
        self.num_heads = num_heads
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, hidden_size)
        
        # Low-rank mLSTM layers
        self.mlstm_cells = nn.ModuleList([
            LowRankMLSTMCell(hidden_size, hidden_size, rank, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_dim)
    
    def forward(self, V: torch.Tensor, t: torch.Tensor,
                states: Optional[list] = None) -> Tuple[torch.Tensor, list]:
        """Forward pass through low-rank mLSTM"""
        x = torch.cat([V, t], dim=1)
        h = self.input_proj(x)
        
        if states is None:
            states = [None] * self.num_layers
        
        new_states = []
        for i, cell in enumerate(self.mlstm_cells):
            h, state = cell(h, states[i])
            new_states.append(state)
        
        I_pred = self.output_proj(h)
        return I_pred, new_states
    
    def count_parameters(self) -> dict:
        """Count parameters and compute compression statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        
        # Calculate theoretical full-rank equivalent
        full_rank_matrix_params = 0
        compressed_matrix_params = 0
        
        for cell in self.mlstm_cells:
            head_dim = cell.head_dim
            num_heads = cell.num_heads
            rank = cell.rank
            
            # Full matrix memory: num_heads × head_dim^2
            full_rank_matrix_params += num_heads * (head_dim ** 2)
            
            # Compressed: 2 × num_heads × head_dim × rank (U and V matrices)
            compressed_matrix_params += 2 * num_heads * head_dim * rank
        
        compression_ratio = compressed_matrix_params / full_rank_matrix_params
        
        return {
            'total_parameters': total_params,
            'matrix_memory_full': full_rank_matrix_params,
            'matrix_memory_compressed': compressed_matrix_params,
            'compression_ratio': compression_ratio,
            'compression_percentage': (1 - compression_ratio) * 100
        }
    
    def get_eigenmode_analysis(self) -> dict:
        """
        Analyze the discovered eigen-modes in low-rank factorization
        Returns the dominant singular values and modes
        """
        eigenmode_analysis = {}
        
        for i, cell in enumerate(self.mlstm_cells):
            U = cell.U.detach().cpu()  # [num_heads, head_dim, rank]
            V = cell.V.detach().cpu()
            
            # For each head, analyze the rank-r subspace
            head_analysis = []
            for h in range(cell.num_heads):
                U_h = U[h, :, :]  # [head_dim, rank]
                V_h = V[h, :, :]
                
                # Compute effective covariance structure: U @ V^T
                effective_cov = U_h @ V_h.T
                
                # SVD to extract dominant modes
                try:
                    _, S, _ = torch.svd(effective_cov)
                    head_analysis.append({
                        'singular_values': S.numpy(),
                        'rank': cell.rank
                    })
                except:
                    head_analysis.append({'singular_values': None, 'rank': cell.rank})
            
            eigenmode_analysis[f'layer_{i}'] = head_analysis
        
        return eigenmode_analysis


if __name__ == "__main__":
    # Test low-rank mLSTM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing LowRankMLSTM:")
    model = LowRankMLSTM(
        input_dim=2,
        hidden_size=64,
        num_layers=2,
        rank=2,
        num_heads=4
    ).to(device)
    
    # Test forward pass
    batch_size = 16
    V = torch.randn(batch_size, 1, device=device)
    t = torch.randn(batch_size, 1, device=device)
    
    I_pred, states = model(V, t)
    print(f"  Input: V{V.shape}, t{t.shape}")
    print(f"  Output: I_pred{I_pred.shape}")
    print(f"  Compressed state shapes: {[s.shape for s in states]}")
    
    # Parameter analysis
    param_stats = model.count_parameters()
    print(f"\n  Parameter Analysis:")
    print(f"    Total parameters: {param_stats['total_parameters']:,}")
    print(f"    Matrix memory (full-rank equivalent): {param_stats['matrix_memory_full']:,}")
    print(f"    Matrix memory (compressed): {param_stats['matrix_memory_compressed']:,}")
    print(f"    Compression ratio: {param_stats['compression_ratio']:.4f}")
    print(f"    Compression: {param_stats['compression_percentage']:.1f}% reduction")
    
    # Eigenmode analysis
    eigenmode_data = model.get_eigenmode_analysis()
    print(f"\n  Eigenmode Analysis:")
    for layer_name, heads in eigenmode_data.items():
        print(f"    {layer_name}:")
        for h, head_data in enumerate(heads):
            if head_data['singular_values'] is not None:
                sv = head_data['singular_values']
                print(f"      Head {h}: σ = {sv[:3]}")  # Top 3 singular values