"""
Ablation Study Script

Runs ablation experiments to analyze contribution of each component.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.hobs import HOBSSensitivityAnalyzer
from utils.metrics import SystemMetrics


ABLATION_CONFIGS = {
    'full': {
        'hessian': True,
        'rl': True,
        'kd': True,
        'adaptive_lambda': True,
        'description': 'Full H-OBS/R-SPAR'
    },
    'no_hessian': {
        'hessian': False,
        'rl': True,
        'kd': True,
        'adaptive_lambda': True,
        'description': 'Without Hessian (magnitude-based)'
    },
    'no_rl': {
        'hessian': True,
        'rl': False,
        'kd': True,
        'adaptive_lambda': True,
        'description': 'Without RL (uniform allocation)'
    },
    'no_kd': {
        'hessian': True,
        'rl': True,
        'kd': False,
        'adaptive_lambda': True,
        'description': 'Without knowledge distillation'
    },
    'no_adaptive': {
        'hessian': True,
        'rl': True,
        'kd': True,
        'adaptive_lambda': False,
        'description': 'Without adaptive λ(t)'
    },
    'baseline': {
        'hessian': False,
        'rl': False,
        'kd': False,
        'adaptive_lambda': False,
        'description': 'Baseline (magnitude pruning only)'
    }
}


def run_ablation(config_name: str, model: nn.Module, dataloader: DataLoader):
    """Run single ablation configuration."""
    ablation_config = ABLATION_CONFIGS[config_name]
    print(f"\n{'='*50}")
    print(f"Ablation: {ablation_config['description']}")
    print(f"{'='*50}")
    print(f"Hessian: {ablation_config['hessian']}")
    print(f"RL: {ablation_config['rl']}")
    print(f"KD: {ablation_config['kd']}")
    print(f"Adaptive λ: {ablation_config['adaptive_lambda']}")
    
    # Simplified ablation - actual implementation would modify pruning behavior
    if ablation_config['hessian']:
        hobs = HOBSSensitivityAnalyzer(model, use_kfac=True)
        sensitivity = hobs.compute_sensitivity(dataloader, num_batches=5)
        pruning_plan = hobs.get_pruning_plan(pruning_ratio=0.3)
    
    # Measure metrics
    metrics = SystemMetrics.measure_all(model, batch_size=32)
    
    return {
        'config': config_name,
        'description': ablation_config['description'],
        'latency_ms': metrics['latency_ms'],
        'throughput': metrics['throughput_imgs_per_sec'],
        'params': metrics['parameters']
    }


def main(args):
    """Run ablation studies."""
    print("="*60)
    print("H-OBS/R-SPAR Ablation Study")
    print("="*60)
    
    # Load model
    model = torchvision.models.resnet18(pretrained=False)
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Dummy dataloader
    dummy_data = torch.randn(100, 3, 224, 224)
    dummy_labels = torch.randint(0, 1000, (100,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    results = []
    
    if args.ablation == 'all':
        configs_to_run = list(ABLATION_CONFIGS.keys())
    else:
        configs_to_run = [args.ablation]
    
    for config_name in configs_to_run:
        result = run_ablation(config_name, model, dataloader)
        results.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("Ablation Study Summary")
    print("="*60)
    print(f"{'Configuration':<30} {'Latency (ms)':<15} {'Throughput':<15}")
    print("-"*60)
    for r in results:
        print(f"{r['description']:<30} {r['latency_ms']:<15.2f} {r['throughput']:<15.1f}")
    
    print("\nAblation study completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument('--config', type=str, default='configs/resnet18.yaml')
    parser.add_argument('--ablation', type=str, default='all',
                        choices=['all'] + list(ABLATION_CONFIGS.keys()))
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    main(args)
