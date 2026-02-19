"""
Time-Constant Clustering Student Network
Implements structure discovery via clustering exponential gates into discrete time constants
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans


class TimeConstantClusteringLSTM(nn.Module):
    """
    LSTM with clustered time constants for forget gates
    Implements equation (5) from manuscript: clustering regularization
    """
    def __init__(self, input_size: int, hidden_size: int, num_clusters: int = 3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_clusters = num_clusters
        
        # Standard LSTM gates
        self.W_gates = nn.Linear(input_size, 4 * hidden_size)
        self.R_gates = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        nn.init.zeros_(self.R_gates.weight)
        
        # Learnable cluster centers μ_k for time constants
        # These represent the dominant physical relaxation times
        # Initialize with diverse values to encourage distinct clusters
        self.cluster_centers = nn.Parameter(
            torch.linspace(-2.0, 2.0, num_clusters).unsqueeze(1).repeat(1, input_size) +
            0.1 * torch.randn(num_clusters, input_size)
        )
        
        # Assignment of hidden units to clusters (initialized randomly)
        self.register_buffer('cluster_assignments', 
                           torch.randint(0, num_clusters, (hidden_size,)))
        
        # Initialize forget gate bias with variety to encourage clustering
        with torch.no_grad():
            self.W_gates.bias[hidden_size:2*hidden_size] = torch.linspace(2.0, 7.0, hidden_size)
    
    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with clustered time constants"""
        batch_size = x.size(0)
        
        if state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            h, c = state
        
        # Compute gates
        gates = self.W_gates(x) + self.R_gates(h)
        i, f_raw, g, o = gates.chunk(4, dim=1)
        
        # Apply sigmoid to forget gate
        # Clustering happens in weight space, not activation space
        f = torch.sigmoid(f_raw)
        
        # Standard LSTM update with clustered/continuous forget gate
        i = torch.exp(i)  # Exponential input gate (xLSTM style)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, (h_new, c_new)
    
    def compute_clustering_loss(self) -> torch.Tensor:
        """
        Compute clustering regularization loss (equation 5)
        Forces forget gate weights to collapse to K cluster centers
        """
        # Get forget gate weights (2nd quarter of gate weights)
        W_f = self.W_gates.weight[self.hidden_size:2*self.hidden_size, :]
        
        # Compute distance from each row to nearest cluster center
        # ||W_f[j] - μ_k||² for all j, k
        cluster_loss = 0.0
        for j in range(self.hidden_size):
            w_j = W_f[j, :]
            # Find distance to nearest cluster
            distances = torch.stack([
                torch.norm(w_j - self.cluster_centers[k]) ** 2
                for k in range(self.num_clusters)
            ])
            cluster_loss += torch.min(distances)
        
        return cluster_loss / self.hidden_size
    
    def update_cluster_assignments(self):
        """
        Update cluster assignments using k-means on forget gate weights
        Called periodically during training
        """
        with torch.no_grad():
            W_f = self.W_gates.weight[self.hidden_size:2*self.hidden_size, :]
            W_f_np = W_f.cpu().numpy()
            
            # K-means clustering with better initialization
            kmeans = KMeans(
                n_clusters=self.num_clusters,
                init='k-means++',  # Better initialization
                random_state=42,
                n_init=20,  # More initialization attempts
                max_iter=300
            )
            
            try:
                labels = kmeans.fit_predict(W_f_np)
                
                # Update assignments and centers (FIX: don't take mean!)
                self.cluster_assignments.copy_(torch.from_numpy(labels))
                self.cluster_centers.data.copy_(torch.from_numpy(kmeans.cluster_centers_))
            except Exception as e:
                print(f"  Warning: K-means clustering failed ({e}), skipping update")
                pass
    
    def get_discovered_time_constants(self) -> np.ndarray:
        """
        Extract discovered physical time constants τ_k
        τ_k = 1 / exp(mean(μ_k)) as per manuscript line 679
        """
        with torch.no_grad():
            # Take mean across input dimensions for each cluster
            mu_k_mean = self.cluster_centers.mean(dim=1)
            tau_k = 1.0 / torch.exp(mu_k_mean)
            return tau_k.cpu().numpy()


class ClusteringStudent(nn.Module):
    """
    Complete student network with time-constant clustering
    Compressed version of xLSTM Teacher
    """
    def __init__(self,
                 input_dim: int = 2,
                 hidden_size: int = 32,  # Smaller than teacher
                 num_layers: int = 2,
                 output_dim: int = 1,
                 num_clusters: int = 3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.num_clusters = num_clusters
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, hidden_size)
        
        # Clustered LSTM layers
        self.lstm_cells = nn.ModuleList([
            TimeConstantClusteringLSTM(hidden_size, hidden_size, num_clusters)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_dim)
    
    def forward(self, V: torch.Tensor, t: torch.Tensor,
                states: Optional[list] = None) -> Tuple[torch.Tensor, list]:
        """Forward pass through student network"""
        x = torch.cat([V, t], dim=1)
        h = self.input_proj(x)
        
        if states is None:
            states = [None] * self.num_layers
        
        new_states = []
        for i, cell in enumerate(self.lstm_cells):
            h, state = cell(h, states[i])
            new_states.append(state)
        
        I_pred = self.output_proj(h)
        return I_pred, new_states
    
    def compute_total_clustering_loss(self) -> torch.Tensor:
        """Compute clustering loss across all layers"""
        total_loss = sum(cell.compute_clustering_loss() for cell in self.lstm_cells)
        return total_loss / self.num_layers
    
    def update_all_clusters(self):
        """Update cluster assignments for all layers"""
        for cell in self.lstm_cells:
            cell.update_cluster_assignments()
    
    def get_all_time_constants(self) -> dict:
        """Extract all discovered time constants"""
        time_constants = {}
        for i, cell in enumerate(self.lstm_cells):
            tau_k = cell.get_discovered_time_constants()
            time_constants[f'layer_{i}'] = tau_k
        return time_constants
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        Count parameters before and after clustering
        
        Returns:
            original: Original parameter count
            compressed: Compressed count (after clustering)
        """
        original = sum(p.numel() for p in self.parameters())
        
        # After clustering: each hidden unit shares K cluster centers
        # instead of unique weights
        compressed = original
        for cell in self.lstm_cells:
            # Forget gate weights: hidden_size rows → K cluster centers
            reduction = cell.hidden_size * cell.input_size - cell.num_clusters
            compressed -= reduction
        
        return original, compressed


if __name__ == "__main__":
    # Test clustering student
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing ClusteringStudent:")
    student = ClusteringStudent(
        input_dim=2,
        hidden_size=32,
        num_layers=2,
        num_clusters=3
    ).to(device)
    
    # Test forward pass
    batch_size = 16
    V = torch.randn(batch_size, 1, device=device)
    t = torch.randn(batch_size, 1, device=device)
    
    I_pred, states = student(V, t)
    print(f"  Input: V{V.shape}, t{t.shape}")
    print(f"  Output: I_pred{I_pred.shape}")
    
    # Test clustering loss
    cluster_loss = student.compute_total_clustering_loss()
    print(f"  Clustering loss: {cluster_loss.item():.6f}")
    
    # Update clusters
    student.update_all_clusters()
    tau_dict = student.get_all_time_constants()
    print(f"  Discovered time constants:")
    for layer, tau_k in tau_dict.items():
        print(f"    {layer}: τ = {tau_k}")
    
    # Parameter count
    orig, comp = student.count_parameters()
    compression = (1 - comp/orig) * 100
    print(f"  Parameters: {orig:,} → {comp:,} ({compression:.1f}% reduction)")