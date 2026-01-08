"""
GPU Profiler Integration

Provides GPU profiling capabilities using NVIDIA tools.
Measures kernel execution time, memory usage, and SM utilization.
"""

import torch
import subprocess
import time
from typing import Dict, Optional
import numpy as np


class GPUProfiler:
    """
    GPU profiler for model performance analysis.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.events_start = []
        self.events_end = []
        self.measurements = {}
    
    def start(self, name: str):
        """Start profiling section."""
        if not self.enabled:
            return
        
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        
        event_start.record()
        self.events_start.append((name, event_start))
        self.events_end.append(event_end)
    
    def end(self, name: str):
        """End profiling section."""
        if not self.enabled:
            return
        
        if self.events_end:
            event_end = self.events_end[-1]
            event_end.record()
            torch.cuda.synchronize()
            
            # Find matching start event
            for i, (n, event_start) in enumerate(self.events_start):
                if n == name:
                    elapsed_time = event_start.elapsed_time(event_end)
                    if name not in self.measurements:
                        self.measurements[name] = []
                    self.measurements[name].append(elapsed_time)
                    break
    
    def get_summary(self) -> Dict:
        """Get profiling summary."""
        summary = {}
        for name, times in self.measurements.items():
            summary[name] = {
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'min_ms': np.min(times),
                'max_ms': np.max(times),
                'count': len(times)
            }
        return summary
    
    def print_summary(self):
        """Print profiling summary."""
        summary = self.get_summary()
        print("\n=== GPU Profiling Summary ===")
        print(f"{'Section':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'Count':<8}")
        print("-" * 65)
        for name, stats in summary.items():
            print(f"{name:<30} {stats['mean_ms']:<12.3f} {stats['std_ms']:<12.3f} {stats['count']:<8}")
    
    def reset(self):
        """Reset profiler."""
        self.events_start = []
        self.events_end = []
        self.measurements = {}


def get_gpu_memory_info() -> Dict:
    """Get GPU memory information."""
    if not torch.cuda.is_available():
        return {}
    
    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
    reserved = torch.cuda.memory_reserved() / 1024**3  # GB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'max_allocated_gb': max_allocated,
        'free_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved
    }


if __name__ == "__main__":
    print("=== GPU Profiler Test ===\n")
    
    if torch.cuda.is_available():
        profiler = GPUProfiler()
        
        # Test profiling
        for i in range(5):
            profiler.start('matrix_multiply')
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            z = torch.matmul(x, y)
            profiler.end('matrix_multiply')
        
        profiler.print_summary()
        
        mem_info = get_gpu_memory_info()
        print("\n=== GPU Memory ===")
        for key, value in mem_info.items():
            print(f"{key}: {value:.2f} GB")
    else:
        print("CUDA not available")
