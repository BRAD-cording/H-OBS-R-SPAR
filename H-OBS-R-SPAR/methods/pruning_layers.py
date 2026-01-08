"""
Structured Pruning Layers

Implements pruning layer wrappers that support dynamic filter removal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import numpy as np


class PrunableConv2d(nn.Module):
    """
    Convolutional layer with pruning support.
    Maintains a mask for pruned filters.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        # Pruning mask (1 = keep, 0 = prune)
        self.register_buffer('mask', torch.ones(out_channels))
        self.pruned_indices = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply mask to weights
        weight = self.conv.weight * self.mask.view(-1, 1, 1, 1)
        bias = self.conv.bias * self.mask if self.conv.bias is not None else None
        
        return F.conv2d(
            x, weight, bias,
            stride=self.conv.stride,
            padding=self.conv.padding
        )
    
    def prune_filters(self, indices: List[int]):
        """Prune specified filter indices."""
        self.pruned_indices.extend(indices)
        with torch.no_grad():
            self.mask[indices] = 0
    
    def get_active_filters(self) -> int:
        """Return number of active (non-pruned) filters."""
        return int(self.mask.sum().item())
    
    def get_compression_ratio(self) -> float:
        """Return compression ratio."""
        total = len(self.mask)
        active = self.get_active_filters()
        return 1.0 - (active / total)


class StructuredPruner:
    """
    Applies structured pruning to a model.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.pruned_filters = {}
    
    def apply_pruning_plan(self, pruning_plan: dict):
        """
        Apply pruning plan to model.
        
        Args:
            pruning_plan: Dict mapping layer names to filter indices to prune
        """
        for name, module in self.model.named_modules():
            if name in pruning_plan and isinstance(module, nn.Conv2d):
                indices = pruning_plan[name]
                self._prune_conv_layer(module, indices)
                self.pruned_filters[name] = indices
    
    def _prune_conv_layer(self, layer: nn.Conv2d, indices: List[int]):
        """Zero out specified filters."""
        with torch.no_grad():
            layer.weight.data[indices] = 0
            if layer.bias is not None:
                layer.bias.data[indices] = 0
    
    def get_statistics(self) -> dict:
        """Get pruning statistics."""
        total_pruned = sum(len(v) for v in self.pruned_filters.values())
        return {
            'total_pruned_filters': total_pruned,
            'num_layers_pruned': len(self.pruned_filters)
        }


if __name__ == "__main__":
    print("=== Pruning Layers Test ===\n")
    
    # Test PrunableConv2d
    layer = PrunableConv2d(64, 128, 3, padding=1)
    x = torch.randn(1, 64, 32, 32)
    
    print(f"Before pruning: {layer.get_active_filters()} filters")
    
    layer.prune_filters([0, 10, 20, 30, 40])
    print(f"After pruning: {layer.get_active_filters()} filters")
    print(f"Compression ratio: {layer.get_compression_ratio():.2%}")
    
    y = layer(x)
    print(f"Output shape: {y.shape}")
