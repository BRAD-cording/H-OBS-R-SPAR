"""
TensorRT Integration Utilities

Converts PyTorch models to TensorRT for optimized inference.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Warning: TensorRT not available. Install tensorrt and pycuda.")


class TensorRTConverter:
    """
    Converts PyTorch models to TensorRT engines.
    """
    def __init__(self, precision: str = 'fp16'):
        """
        Args:
            precision: 'fp32', 'fp16', or 'int8'
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")
        
        self.precision = precision
        self.logger = trt.Logger(trt.Logger.WARNING)
    
    def convert_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        max_batch_size: int = 64
    ) -> str:
        """
        Convert PyTorch model to TensorRT engine.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            max_batch_size: Maximum batch size
        
        Returns:
            Path to serialized TensorRT engine
        """
        # Export to ONNX first
        onnx_path = "model.onnx"
        
        dummy_input = torch.randn(*input_shape)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # Build TensorRT engine
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("ERROR: Failed to parse ONNX")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        if self.precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        # Serialize engine
        engine_path = f"model_{self.precision}.trt"
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"TensorRT engine saved to: {engine_path}")
        return engine_path


class TensorRTBenchmark:
    """
    Benchmark TensorRT engine performance.
    """
    def __init__(self, engine_path: str):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
    
    def benchmark(
        self,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        num_iterations: int = 100
    ) -> dict:
        """
        Benchmark TensorRT engine.
        
        Returns:
            Dictionary with latency and throughput
        """
        import time
        
        # Allocate buffers
        batch_size = input_shape[0]
        input_size = np.prod(input_shape) * np.dtype(np.float32).itemsize
        output_size = batch_size * 1000 * np.dtype(np.float32).itemsize
        
        h_input = cuda.pagelocked_empty(np.prod(input_shape), dtype=np.float32)
        h_output = cuda.pagelocked_empty(batch_size * 1000, dtype=np.float32)
        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)
        
        stream = cuda.Stream()
        
        # Warmup
        for _ in range(10):
            cuda.memcpy_htod_async(d_input, h_input, stream)
            self.context.execute_async_v2(
                bindings=[int(d_input), int(d_output)],
                stream_handle=stream.handle
            )
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
        
        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            
            cuda.memcpy_htod_async(d_input, h_input, stream)
            self.context.execute_async_v2(
                bindings=[int(d_input), int(d_output)],
                stream_handle=stream.handle
            )
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            stream.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
        
        latency_mean = np.mean(latencies)
        latency_std = np.std(latencies)
        throughput = (batch_size * num_iterations) / (sum(latencies) / 1000)
        
        return {
            'latency_ms': latency_mean,
            'latency_std_ms': latency_std,
            'throughput_imgs_per_sec': throughput
        }


if __name__ == "__main__":
    print("=== TensorRT Utils Test ===\n")
    
    if TRT_AVAILABLE:
        import torchvision
        
        model = torchvision.models.resnet18(pretrained=False)
        model.eval()
        
        print("Converting model to TensorRT...")
        converter = TensorRTConverter(precision='fp16')
        
        try:
            engine_path = converter.convert_model(model)
            
            if engine_path:
                print("\nBenchmarking TensorRT engine...")
                benchmark = TensorRTBenchmark(engine_path)
                results = benchmark.benchmark()
                
                print(f"\nResults:")
                print(f"  Latency: {results['latency_ms']:.2f} ± {results['latency_std_ms']:.2f} ms")
                print(f"  Throughput: {results['throughput_imgs_per_sec']:.1f} images/s")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("TensorRT not available. Skipping test.")
