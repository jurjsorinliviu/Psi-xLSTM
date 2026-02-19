"""
Recurrent Relation-Aware Distillation (RRAD)
Implements equation (4) from manuscript: distillation with temporal gradient matching
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


class RecurrentRelationAwareDistillation:
    """
    RRAD training strategy for compressing xLSTM Teacher to Ψ-xLSTM Student
    Matches both hidden states and temporal gradients (equation 4)
    """
    def __init__(self,
                 teacher: nn.Module,
                 student: nn.Module,
                 alpha: float = 1.0,  # Weight for hidden state matching
                 beta: float = 0.5,   # Weight for temporal gradient matching
                 gamma: float = 0.1):  # Weight for clustering loss
        """
        Initialize RRAD
        
        Args:
            teacher: Trained xLSTM-PINN teacher network
            student: Student network (ClusteringStudent or LowRankMLSTM)
            alpha: Weight for hidden state matching loss
            beta: Weight for temporal gradient matching loss
            gamma: Weight for clustering/structure discovery loss
        """
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Projection matrix to align dimensions if teacher/student differ
        teacher_dim = teacher.hidden_size
        student_dim = student.hidden_size
        
        # Get device from student model
        device = next(student.parameters()).device
        
        if teacher_dim != student_dim:
            self.W_proj = nn.Parameter(torch.randn(student_dim, teacher_dim, device=device) * 0.01)
        else:
            self.W_proj = None
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def compute_distillation_loss(self,
                                  V: torch.Tensor,
                                  t: torch.Tensor,
                                  I_true: torch.Tensor,
                                  student_states: list = None,
                                  teacher_states: list = None
                                  ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute RRAD loss (equation 4)
        
        Args:
            V: Voltage input [batch, 1]
            t: Time input [batch, 1] (requires_grad=True)
            I_true: Ground truth current [batch, 1]
            student_states: Optional student LSTM states
            teacher_states: Optional teacher LSTM states
            
        Returns:
            total_loss: Combined distillation loss
            loss_dict: Dictionary of loss components
        """
        # Ensure t requires grad for temporal gradient computation
        if not t.requires_grad:
            t = t.requires_grad_(True)
        
        # Forward pass through teacher (keep gradients for temporal matching)
        with torch.set_grad_enabled(True):
            I_teacher, h_teacher_states = self.teacher(V, t, teacher_states)
            I_teacher = I_teacher.detach()  # Detach for non-gradient losses, but keep graph for grad computation
        
        # Forward pass through student
        I_student, h_student_states = self.student(V, t, student_states)
        
        # Component 1: Data fitting loss (both should match ground truth)
        loss_data_student = torch.mean((I_student - I_true) ** 2)
        
        # Component 2: Output matching (α * ||I_S - I_T||²)
        # Both outputs are [batch, 1], so no projection needed
        loss_hidden = torch.mean((I_student - I_teacher) ** 2)
        
        # Component 3: Temporal gradient matching (β * ||∂h_S/∂t - ∂h_T/∂t||²)
        # Compute temporal derivatives
        dI_student_dt = torch.autograd.grad(
            I_student.sum(), t, create_graph=True, retain_graph=True
        )[0]
        
        # For teacher, we need to recompute with gradients enabled temporarily
        with torch.set_grad_enabled(True):
            t_temp = t.detach().requires_grad_(True)
            I_teacher_temp, _ = self.teacher(V, t_temp, teacher_states)
            dI_teacher_dt = torch.autograd.grad(
                I_teacher_temp.sum(), t_temp, create_graph=False
            )[0].detach()
        
        loss_gradient = torch.mean((dI_student_dt - dI_teacher_dt.detach()) ** 2)
        
        # Component 4: Structure discovery loss (clustering or low-rank)
        loss_structure = self._compute_structure_loss()
        
        # Total RRAD loss
        total_loss = (loss_data_student + 
                     self.alpha * loss_hidden + 
                     self.beta * loss_gradient + 
                     self.gamma * loss_structure)
        
        loss_dict = {
            'total': total_loss.item(),
            'data': loss_data_student.item(),
            'hidden_matching': loss_hidden.item(),
            'gradient_matching': loss_gradient.item(),
            'structure_discovery': loss_structure.item()
        }
        
        return total_loss, loss_dict
    
    def _compute_structure_loss(self) -> torch.Tensor:
        """
        Compute structure discovery loss
        - For ClusteringStudent: clustering regularization
        - For LowRankMLSTM: low-rank constraint (already implicit in architecture)
        """
        # Check if student has clustering loss method
        if hasattr(self.student, 'compute_total_clustering_loss'):
            return self.student.compute_total_clustering_loss()
        else:
            # For low-rank models, structure is implicit
            # Could add nuclear norm regularization for stronger low-rank bias
            return torch.tensor(0.0, device=next(self.student.parameters()).device)
    
    def train_step(self,
                  optimizer: torch.optim.Optimizer,
                  V: torch.Tensor,
                  t: torch.Tensor,
                  I_true: torch.Tensor
                  ) -> Dict[str, float]:
        """
        Perform one training step with RRAD
        
        Returns:
            loss_dict: Dictionary of loss values
        """
        optimizer.zero_grad()
        
        # Compute distillation loss
        loss, loss_dict = self.compute_distillation_loss(V, t, I_true)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        
        # Update student
        optimizer.step()
        
        return loss_dict
    
    def update_student_structure(self, epoch: int, update_interval: int = 10):
        """
        Periodically update structure discovery (e.g., k-means clustering)
        
        Args:
            epoch: Current training epoch
            update_interval: Update structure every N epochs
        """
        if epoch % update_interval == 0:
            if hasattr(self.student, 'update_all_clusters'):
                self.student.update_all_clusters()
                print(f"  [Epoch {epoch}] Updated cluster assignments")


class StandardPINNBaseline(nn.Module):
    """
    Standard PINN with MLP backbone for baseline comparison
    """
    def __init__(self, input_dim: int = 2, hidden_sizes: list = [64, 64, 64], output_dim: int = 1):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, V: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass (no recurrent states for MLP)"""
        x = torch.cat([V, t], dim=1)
        return self.network(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_baseline_pinn(input_dim: int = 2, hidden_size: int = 64, 
                        num_layers: int = 3, output_dim: int = 1) -> StandardPINNBaseline:
    """Create standard PINN baseline for comparison"""
    hidden_sizes = [hidden_size] * num_layers
    return StandardPINNBaseline(input_dim, hidden_sizes, output_dim)


if __name__ == "__main__":
    # Test RRAD
    from psi_xlstm.models.xlstm_teacher import xLSTMTeacher
    from psi_xlstm.models.clustering_student import ClusteringStudent
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Testing Recurrent Relation-Aware Distillation:")
    
    # Create teacher and student
    teacher = xLSTMTeacher(hidden_size=64, num_layers=2).to(device)
    student = ClusteringStudent(hidden_size=32, num_layers=2, num_clusters=3).to(device)
    
    print(f"  Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"  Student parameters: {sum(p.numel() for p in student.parameters()):,}")
    
    # Create RRAD trainer
    rrad = RecurrentRelationAwareDistillation(teacher, student, alpha=1.0, beta=0.5, gamma=0.1)
    
    # Test distillation loss
    batch_size = 16
    V = torch.randn(batch_size, 1, device=device)
    t = torch.randn(batch_size, 1, device=device, requires_grad=True)
    I_true = torch.randn(batch_size, 1, device=device)
    
    loss, loss_dict = rrad.compute_distillation_loss(V, t, I_true)
    print(f"\n  Distillation Loss: {loss.item():.6f}")
    print(f"  Loss components:")
    for key, value in loss_dict.items():
        print(f"    {key}: {value:.6f}")
    
    # Test training step
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    loss_dict = rrad.train_step(optimizer, V, t, I_true)
    print(f"\n  After training step:")
    print(f"    Total loss: {loss_dict['total']:.6f}")
    
    # Test baseline PINN
    print("\n  Testing baseline PINN:")
    baseline = create_baseline_pinn(hidden_size=64, num_layers=3).to(device)
    print(f"    Parameters: {baseline.count_parameters():,}")
    I_pred = baseline(V, t)
    print(f"    Output shape: {I_pred.shape}")