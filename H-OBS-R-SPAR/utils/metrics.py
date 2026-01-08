"""
System-level performance metrics for model evaluation.

Metrics include:
- Latency (ms/image)
- Throughput (images/s)
- Power consumption (W)
- Energy per inference (J/image)
- FLOPs and Parameters
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Tuple, Optional
import subprocess


class SystemMetrics:
    """
    Comprehensive system-level metrics measurement.
    """
    
    @staticmethod
    def measure_latency(
        model: nn.Module,
        input_shape: Tuple = (1, 3, 224, 224),
        num_iterations: int = 100,
        warmup: int = 10
    ) -> Tuple[float, float]:
        """
        Measure inference latency.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            num_iterations: Number of inference iterations
            warmup: Warmup iterations
        
        Returns:
            (mean_latency_ms, std_latency_ms)
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                x = torch.randn(*input_shape, device=device)
                _ = model(x)
        
        # Measure
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                x = torch.randn(*input_shape, device=device)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = model(x)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                else:
                    start = time.perf_counter()
                    _ = model(x)
                    end = time.perf_counter()
                
                latency_ms = (end - start) * 1000
                latencies.append(latency_ms)
        
        return np.mean(latencies), np.std(latencies)
    
    @staticmethod
    def measure_throughput(
        model: nn.Module,
        batch_size: int = 64,
        input_shape: Tuple = (3, 224, 224),
        num_iterations: int = 50
    ) -> float:
        """
        Measure throughput (images/second).
        
        Args:
            model: PyTorch model
            batch_size: Batch size
            input_shape: Single image shape
            num_iterations: Number of iterations
        
        Returns:
            throughput (images/s)
        """
        model.eval()
        device = next(model.parameters()).device
        
        total_images = 0
        total_time = 0
        
        with torch.no_grad():
            for _ in range(num_iterations):
                x = torch.randn(batch_size, *input_shape, device=device)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                
                total_images += batch_size
                total_time += (end - start)
        
        throughput = total_images / total_time
        return throughput
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def count_flops(model: nn.Module, input_shape: Tuple = (1, 3, 224, 224)) -> int:
        """
        Estimate FLOPs (simplified).
        """
        total_flops = 0
        
        def count_conv_flops(module, input, output):
            nonlocal total_flops
            if isinstance(module, nn.Conv2d):
                # FLOPs = K * K * Cin * Cout * H * W / groups
                batch_size = output.size(0)
                output_h = output.size(2)
                output_w = output.size(3)
                kernel_ops = module.kernel_size[0] * module.kernel_size[1] * \
                             (module.in_channels // module.groups)
                flops = kernel_ops * module.out_channels * output_h * output_w * batch_size
                total_flops += flops
            elif isinstance(module, nn.Linear):
                flops = module.in_features * module.out_features * input[0].size(0)
                total_flops += flops
        
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(count_conv_flops))
        
        device = next(model.parameters()).device
        x = torch.randn(*input_shape, device=device)
        model.eval()
        with torch.no_grad():
            model(x)
        
        for hook in hooks:
            hook.remove()
        
        return total_flops
    
    @staticmethod
    def get_model_size_mb(model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    @staticmethod
    def measure_all(
        model: nn.Module,
        batch_size: int = 64,
        input_shape: Tuple = (3, 224, 224)
    ) -> Dict:
        """
        Measure all metrics.
        
        Returns:
            Dictionary with all metrics
        """
        print("Measuring system metrics...")
        
        latency_mean, latency_std = SystemMetrics.measure_latency(
            model, 
            input_shape=(1, *input_shape)
        )
        
        throughput = SystemMetrics.measure_throughput(
            model,
            batch_size=batch_size,
            input_shape=input_shape
        )
        
        params = SystemMetrics.count_parameters(model)
        flops = SystemMetrics.count_flops(model, input_shape=(1, *input_shape))
        model_size = SystemMetrics.get_model_size_mb(model)
        
        metrics = {
            'latency_ms': latency_mean,
            'latency_std_ms': latency_std,
            'throughput_imgs_per_sec': throughput,
            'parameters': params,
            'flops': flops,
            'model_size_mb': model_size,
            'speedup': 1.0,  # Will be computed relative to baseline
            'energy_per_image_j': latency_mean * 0.001 * 200,  # Estimated
        }
        
        return metrics


def print_metrics(metrics: Dict, title: str = "Model Metrics"):
    """Pretty print metrics."""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(f"Latency:       {metrics['latency_ms']:.2f} ± {metrics['latency_std_ms']:.2f} ms")
    print(f"Throughput:    {metrics['throughput_imgs_per_sec']:.1f} images/s")
    print(f"Parameters:    {metrics['parameters']:,}")
    print(f"FLOPs:         {metrics['flops']:,}")
    print(f"Model Size:    {metrics['model_size_mb']:.2f} MB")
    print(f"Energy/Image:  {metrics['energy_per_image_j']:.2f} J")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import torchvision.models as models
    
    print("=== System Metrics Test ===\n")
    
    model = models.resnet18(pretrained=False)
    if torch.cuda.is_available():
        model = model.cuda()
    
    metrics = SystemMetrics.measure_all(model, batch_size=32)
    print_metrics(metrics, "ResNet-18 Metrics")
