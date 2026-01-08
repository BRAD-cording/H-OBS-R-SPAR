"""
Cross-Platform Validation Script

Tests model performance across different GPU platforms.
"""

import argparse
import yaml
import torch
import torchvision
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import SystemMetrics, print_metrics


PLATFORMS = {
    'rtx5090': {'name': 'RTX-5090', 'compute_capability': (8, 9)},
    'h100': {'name': 'H100', 'compute_capability': (9, 0)},
    'a100': {'name': 'A100', 'compute_capability': (8, 0)},
    'v100': {'name': 'V100', 'compute_capability': (7, 0)},
}


def detect_gpu_platform() -> str:
    """Detect current GPU platform."""
    if not torch.cuda.is_available():
        return 'cpu'
    
    gpu_name = torch.cuda.get_device_name(0).lower()
    
    if 'h100' in gpu_name:
        return 'h100'
    elif 'a100' in gpu_name:
        return 'a100'
    elif 'v100' in gpu_name:
        return 'v100'
    elif '5090' in gpu_name or 'rtx 5090' in gpu_name:
        return 'rtx5090'
    else:
        return 'unknown'


def main(args):
    """Run cross-platform validation."""
    print("="*70)
    print("Cross-Platform Validation")
    print("="*70)
    
    # Detect platform
    platform = detect_gpu_platform()
    print(f"\nDetected Platform: {platform}")
    
    if platform != 'cpu' and platform != 'unknown':
        platform_info = PLATFORMS.get(platform, {})
        print(f"Platform Name: {platform_info.get('name', 'Unknown')}")
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    model_name = config['model']['name']
    print(f"\nModel: {model_name}")
    
    if model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False)
    elif model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
    elif model_name == 'mobilenetv2':
        model = torchvision.models.mobilenet_v2(pretrained=False)
    else:
        model = torchvision.models.efficientnet_b0(pretrained=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Measure metrics
    print("\nMeasuring performance...")
    metrics = SystemMetrics.measure_all(model, batch_size=args.batch_size)
    
    print_metrics(metrics, f"{platform.upper()} - {model_name.upper()}")
    
    # Save results
    import json
    output_file = f"results/{platform}_{model_name}_metrics.json"
    Path("results").mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'platform': platform,
            'model': model_name,
            'metrics': metrics
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Platform Validation")
    parser.add_argument('--config', type=str, default='configs/resnet50.yaml')
    parser.add_argument('--batch-size', type=int, default=64)
    
    args = parser.parse_args()
    main(args)
