"""
TensorRT Benchmark Script

Benchmarks TensorRT optimized models.
"""

import argparse
import yaml
import torch
import torchvision
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tensorrt_utils import TensorRTConverter, TensorRTBenchmark, TRT_AVAILABLE
from utils.metrics import SystemMetrics


def main(args):
    """Run TensorRT benchmark."""
    print("="*70)
    print("TensorRT Benchmark")
    print("="*70)
    
    if not TRT_AVAILABLE:
        print("\nError: TensorRT not available")
        print("Install: pip install tensorrt pycuda")
        return
    
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
    
    model.eval()
    
    # Convert to TensorRT
    print(f"\nConverting to TensorRT ({args.precision})...")
    converter = TensorRTConverter(precision=args.precision)
    
    try:
        engine_path = converter.convert_model(
            model,
            input_shape=(args.batch_size, 3, 224, 224),
            max_batch_size=args.batch_size
        )
        
        if not engine_path:
            print("Conversion failed")
            return
        
        # Benchmark TensorRT
        print(f"\nBenchmarking TensorRT engine...")
        trt_benchmark = TensorRTBenchmark(engine_path)
        trt_results = trt_benchmark.benchmark(
            input_shape=(args.batch_size, 3, 224, 224),
            num_iterations=args.num_iterations
        )
        
        # Benchmark PyTorch for comparison
        print("Benchmarking PyTorch...")
        model_cuda = model.cuda()
        pytorch_results = SystemMetrics.measure_all(model_cuda, batch_size=args.batch_size)
        
        # Print comparison
        print("\n" + "="*70)
        print("Performance Comparison")
        print("="*70)
        print(f"{'Metric':<30} {'PyTorch':<20} {'TensorRT':<20} {'Speedup':<15}")
        print("-"*85)
        
        pytorch_latency = pytorch_results['latency_ms']
        trt_latency = trt_results['latency_ms']
        speedup = pytorch_latency / trt_latency
        
        print(f"{'Latency (ms)':<30} {pytorch_latency:<20.2f} {trt_latency:<20.2f} {speedup:<15.2f}×")
        
        pytorch_throughput = pytorch_results['throughput_imgs_per_sec']
        trt_throughput = trt_results['throughput_imgs_per_sec']
        throughput_gain = trt_throughput / pytorch_throughput
        
        print(f"{'Throughput (img/s)':<30} {pytorch_throughput:<20.1f} {trt_throughput:<20.1f} {throughput_gain:<15.2f}×")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT Benchmark")
    parser.add_argument('--config', type=str, default='configs/resnet50.yaml')
    parser.add_argument('--precision', type=str, default='fp16', choices=['fp32', 'fp16', 'int8'])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-iterations', type=int, default=100)
    
    args = parser.parse_args()
    main(args)
