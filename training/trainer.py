"""
Training Module for xLSTM-PINN Teacher and Ψ-xLSTM Student
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm
import time
import os


def train_teacher(teacher: nn.Module,
                 dataset: dict,
                 num_epochs: int = 100,
                 batch_size: int = 256,
                 learning_rate: float = 1e-3,
                 device: str = 'cuda',
                 lambda_data: float = 1.0,
                 lambda_pde: float = 0.1,
                 lambda_ic: float = 0.1,
                 save_dir: str = '.') -> Tuple[nn.Module, dict]:
    """
    Train xLSTM-PINN teacher network
    
    Args:
        teacher: xLSTM Teacher model
        dataset: Dictionary with 'train', 'val' splits
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        lambda_data: Weight for data loss
        lambda_pde: Weight for physics loss
        lambda_ic: Weight for initial condition loss
        
    Returns:
        trained_teacher: Trained model
        history: Training history dictionary
    """
    print(f"\n{'='*60}")
    print("Training xLSTM-PINN Teacher Network")
    print(f"{'='*60}")
    
    teacher = teacher.to(device)
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Create dataloaders
    train_dataset = TensorDataset(
        dataset['train']['V'], dataset['train']['t'], dataset['train']['I']
    )
    val_dataset = TensorDataset(
        dataset['val']['V'], dataset['val']['t'], dataset['val']['I']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_data_loss': [],
        'train_pde_loss': [],
        'train_ic_loss': [],
        'epoch_time': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        teacher.train()
        train_losses = []
        train_data_losses = []
        train_pde_losses = []
        train_ic_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for V_batch, t_batch, I_batch in pbar:
            t_batch = t_batch.requires_grad_(True)  # Enable gradient for PDE loss
            
            optimizer.zero_grad()
            
            # Forward pass
            I_pred, _ = teacher(V_batch, t_batch)
            
            # Compute physics-informed loss
            loss, loss_dict = teacher.compute_physics_loss(
                V_batch, t_batch, I_pred, I_batch, {},
                lambda_data, lambda_pde, lambda_ic
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss_dict['total'])
            train_data_losses.append(loss_dict['data'])
            train_pde_losses.append(loss_dict['pde'])
            train_ic_losses.append(loss_dict['ic'])
            
            pbar.set_postfix({'loss': f"{loss_dict['total']:.6f}"})
        
        avg_train_loss = np.mean(train_losses)
        avg_train_data = np.mean(train_data_losses)
        avg_train_pde = np.mean(train_pde_losses)
        avg_train_ic = np.mean(train_ic_losses)
        
        # Validation phase
        teacher.eval()
        val_losses = []
        
        with torch.no_grad():
            for V_batch, t_batch, I_batch in val_loader:
                I_pred, _ = teacher(V_batch, t_batch)
                val_loss = torch.mean((I_pred - I_batch) ** 2)
                val_losses.append(val_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, 'teacher_best.pth')
            torch.save(teacher.state_dict(), save_path)
        
        epoch_time = time.time() - epoch_start
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_data_loss'].append(avg_train_data)
        history['train_pde_loss'].append(avg_train_pde)
        history['train_ic_loss'].append(avg_train_ic)
        history['epoch_time'].append(epoch_time)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train={avg_train_loss:.3e}, Val={avg_val_loss:.3e}, "
                  f"Data={avg_train_data:.3e}, PDE={avg_train_pde:.3e}, "
                  f"Time={epoch_time:.2f}s")
    
    print(f"\nTeacher training completed!")
    print(f"Best validation loss: {best_val_loss:.3e}")
    
    # Load best model
    load_path = os.path.join(save_dir, 'teacher_best.pth')
    teacher.load_state_dict(torch.load(load_path))
    
    return teacher, history


def train_student(student: nn.Module,
                 teacher: nn.Module,
                 dataset: dict,
                 num_epochs: int = 150,
                 batch_size: int = 256,
                 learning_rate: float = 1e-3,
                 device: str = 'cuda',
                 alpha: float = 1.0,
                 beta: float = 0.5,
                 gamma: float = 0.1,
                 update_clusters_interval: int = 10,
                 save_dir: str = '.') -> Tuple[nn.Module, dict]:
    """
    Train Ψ-xLSTM student with RRAD
    
    Args:
        student: Student model (ClusteringStudent or LowRankMLSTM)
        teacher: Trained teacher model
        dataset: Training dataset
        num_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Training device
        alpha: Hidden state matching weight
        beta: Gradient matching weight
        gamma: Structure discovery weight
        update_clusters_interval: Cluster update frequency
        
    Returns:
        trained_student: Trained student model
        history: Training history
    """
    print(f"\n{'='*60}")
    print("Training Ψ-xLSTM Student with RRAD")
    print(f"{'='*60}")
    
    from psi_xlstm.training.distillation import RecurrentRelationAwareDistillation
    
    student = student.to(device)
    teacher = teacher.to(device)
    
    # Create RRAD trainer
    rrad = RecurrentRelationAwareDistillation(teacher, student, alpha, beta, gamma)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Create dataloader
    train_dataset = TensorDataset(
        dataset['train']['V'], dataset['train']['t'], dataset['train']['I']
    )
    val_dataset = TensorDataset(
        dataset['val']['V'], dataset['val']['t'], dataset['val']['I']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'hidden_matching_loss': [],
        'gradient_matching_loss': [],
        'structure_loss': [],
        'epoch_time': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        student.train()
        epoch_losses = {
            'total': [],
            'data': [],
            'hidden_matching': [],
            'gradient_matching': [],
            'structure_discovery': []
        }
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for V_batch, t_batch, I_batch in pbar:
            t_batch = t_batch.requires_grad_(True)
            
            loss_dict = rrad.train_step(optimizer, V_batch, t_batch, I_batch)
            
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)
            
            pbar.set_postfix({'loss': f"{loss_dict['total']:.6f}"})
        
        # Update cluster assignments if applicable
        rrad.update_student_structure(epoch, update_clusters_interval)
        
        # Validation
        student.eval()
        val_losses = []
        
        with torch.no_grad():
            for V_batch, t_batch, I_batch in val_loader:
                I_pred, _ = student(V_batch, t_batch)
                val_loss = torch.mean((I_pred - I_batch) ** 2)
                val_losses.append(val_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        scheduler.step()
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, 'student_best.pth')
            torch.save(student.state_dict(), save_path)
        
        epoch_time = time.time() - epoch_start
        
        # Store history
        avg_train_loss = np.mean(epoch_losses['total'])
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['hidden_matching_loss'].append(np.mean(epoch_losses['hidden_matching']))
        history['gradient_matching_loss'].append(np.mean(epoch_losses['gradient_matching']))
        history['structure_loss'].append(np.mean(epoch_losses['structure_discovery']))
        history['epoch_time'].append(epoch_time)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train={avg_train_loss:.3e}, Val={avg_val_loss:.3e}, "
                  f"Hidden={np.mean(epoch_losses['hidden_matching']):.3e}, "
                  f"Grad={np.mean(epoch_losses['gradient_matching']):.3e}, "
                  f"Time={epoch_time:.2f}s")
    
    print(f"\nStudent training completed!")
    print(f"Best validation loss: {best_val_loss:.3e}")
    
    # Load best model
    load_path = os.path.join(save_dir, 'student_best.pth')
    student.load_state_dict(torch.load(load_path))
    
    return student, history


if __name__ == "__main__":
    # Test training functions
    print("Testing training functions...")