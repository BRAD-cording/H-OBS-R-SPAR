"""
System-level evaluation script.

Evaluates pruned models on latency, throughput, energy, and accuracy.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import SystemMetrics, print_metrics


def main(args):
    """Main evaluation function."""
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Evaluating: {args.checkpoint}")
    
    # Load model
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    model_name = config['model']['name']
    if model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False)
    elif model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
    elif model_name == 'mobilenetv2':
        model = torchvision.models.mobilenet_v2(pretrained=False)
    else:
        model = torchvision.models.efficientnet_b0(pretrained=False)
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        print("Loaded checkpoint successfully")
    
    model = model.to(device)
    model.eval()
    
    # Measure metrics
    print("\nMeasuring system metrics...")
    metrics = SystemMetrics.measure_all(
        model,
        batch_size=args.batch_size,
        input_shape=(3, 224, 224)
    )
    
    print_metrics(metrics, f"{model_name.upper()} System Metrics")
    
    # Save results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System-level Evaluation")
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    
    args = parser.parse_args()
    main(args)
