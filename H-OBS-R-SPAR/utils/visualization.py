"""
Visualization Tools

Creates plots for training curves, feature maps, and performance analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import seaborn as sns


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot training and validation curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    if 'train_acc' in history:
        ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
    if 'val_acc' in history:
        ax2.plot(history['val_acc'], label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_pruning_comparison(
    methods: List[str],
    accuracies: List[float],
    speedups: List[float],
    save_path: Optional[str] = None
):
    """
    Plot comparison of pruning methods.
    
    Args:
        methods: List of method names
        accuracies: List of accuracies
        speedups: List of speedup ratios
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    colors = sns.color_palette("husl", len(methods))
    ax1.barh(methods, accuracies, color=colors)
    ax1.set_xlabel('Top-1 Accuracy (%)')
    ax1.set_title('Accuracy Comparison')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Speedup comparison
    ax2.barh(methods, speedups, color=colors)
    ax2.set_xlabel('Speedup (×)')
    ax2.set_title('Speedup Comparison')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_sensitivity_heatmap(
    sensitivity_scores: np.ndarray,
    layer_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot heatmap of filter sensitivity scores.
    
    Args:
        sensitivity_scores: 2D array [layers, filters]
        layer_names: List of layer names
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 8))
    
    sns.heatmap(
        sensitivity_scores,
        yticklabels=layer_names,
        cmap='viridis',
        cbar_kws={'label': 'Sensitivity Score'}
    )
    
    plt.xlabel('Filter Index')
    plt.ylabel('Layer')
    plt.title('Filter Sensitivity Heatmap')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_lambda_evolution(
    lambda_history: List[float],
    acc_history: List[float],
    save_path: Optional[str] = None
):
    """
    Plot adaptive λ(t) evolution over training.
    
    Args:
        lambda_history: List of lambda values
        acc_history: List of accuracy values
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Lambda evolution
    ax1.plot(lambda_history, linewidth=2, color='#2E86AB')
    ax1.set_ylabel('λ(t)')
    ax1.set_title('Adaptive Regularization λ(t) Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy evolution
    ax2.plot(acc_history, linewidth=2, color='#A23B72')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy During Training')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("=== Visualization Tools Test ===\n")
    
    # Test training curves
    history = {
        'train_loss': np.random.rand(100) * 2 + 0.5,
        'val_loss': np.random.rand(100) * 2 + 0.7,
        'train_acc': np.random.rand(100) * 10 + 70,
        'val_acc': np.random.rand(100) * 10 + 68
    }
    
    print("Plotting training curves...")
    plot_training_curves(history)
    
    # Test method comparison
    methods = ['Unpruned', 'DepGraph', 'HRank', 'H-OBS/R-SPAR']
    accuracies = [76.13, 75.82, 75.65, 75.96]
    speedups = [1.0, 3.03, 2.85, 3.10]
    
    print("Plotting method comparison...")
    plot_pruning_comparison(methods, accuracies, speedups)
    
    print("\nVisualization test completed!")
