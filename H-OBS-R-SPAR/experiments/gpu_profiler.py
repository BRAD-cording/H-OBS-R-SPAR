"""
GPU Profiling Script

Detailed GPU profiling including kernel time and memory analysis.
"""

import argparse
import yaml
import torch
import torchvision
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.gpu_profiler import GPUProfiler, get_gpu_memory_info
from utils.metrics import SystemMetrics


def main(args):
    """Run GPU profiling."""
    print("="*70)
    print("GPU Profiling")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\nError: CUDA not available")
        return
    
    # Print GPU info
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    
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
    
    model = model.cuda()
    model.eval()
    
    # Initialize profiler
    profiler = GPUProfiler(enabled=True)
    
    # Profile inference
    print(f"\nProfiling {args.num_iterations} iterations...")
    
    with torch.no_grad():
        for i in range(args.num_iterations):
            profiler.start('forward_pass')
            
            x = torch.randn(args.batch_size, 3, 224, 224, device='cuda')
            output = model(x)
            
            profiler.end('forward_pass')
    
    # Print profiling results
    profiler.print_summary()
    
    # Print memory info
    mem_info = get_gpu_memory_info()
    print("\n=== GPU Memory Usage ===")
    for key, value in mem_info.items():
        print(f"{key}: {value:.2f} GB")
    
    # Additional metrics
    print("\n=== Additional Metrics ===")
    metrics = SystemMetrics.measure_all(model, batch_size=args.batch_size)
    print(f"Latency: {metrics['latency_ms']:.2f} ms")
    print(f"Throughput: {metrics['throughput_imgs_per_sec']:.1f} images/s")
    print(f"Model Size: {metrics['model_size_mb']:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Profiling")
    parser.add_argument('--config', type=str, default='configs/resnet50.yaml')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-iterations', type=int, default=100)
    
    args = parser.parse_args()
    main(args)
