"""
Batch Size Analysis Script

Analyzes performance across different batch sizes.
"""

import argparse
import yaml
import torch
import torchvision
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import SystemMetrics


def main(args):
    """Analyze performance across batch sizes."""
    print("="*70)
    print("Batch Size Analysis")
    print("="*70)
    
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
    
    # Test different batch sizes
    batch_sizes = args.batch_sizes or [16, 32, 64, 128]
    
    print(f"\nTesting batch sizes: {batch_sizes}")
    print("\n" + "="*90)
    print(f"{'Batch Size':<12} {'Latency (ms)':<15} {'Throughput':<15} {'Memory (GB)':<15}")
    print("-"*90)
    
    results = []
    
    for bs in batch_sizes:
        try:
            # Measure latency
            latency_mean, latency_std = SystemMetrics.measure_latency(
                model,
                input_shape=(1, 3, 224, 224),
                num_iterations=50
            )
            
            # Measure throughput
            throughput = SystemMetrics.measure_throughput(
                model,
                batch_size=bs,
                input_shape=(3, 224, 224),
                num_iterations=30
            )
            
            # Memory usage
            if torch.cuda.is_available():
                memory_gb = torch.cuda.max_memory_allocated() / 1024**3
            else:
                memory_gb = 0
            
            print(f"{bs:<12} {latency_mean:<15.2f} {throughput:<15.1f} {memory_gb:<15.2f}")
            
            results.append({
                'batch_size': bs,
                'latency_ms': latency_mean,
                'throughput': throughput,
                'memory_gb': memory_gb
            })
            
            # Reset memory stats
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                
        except RuntimeError as e:
            print(f"{bs:<12} OOM - Out of Memory")
            break
    
    print("="*90)
    
    # Save results
    import json
    output_file = f"results/batch_analysis_{model_name}.json"
    Path("results").mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Size Analysis")
    parser.add_argument('--config', type=str, default='configs/resnet50.yaml')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=None)
    
    args = parser.parse_args()
    main(args)
