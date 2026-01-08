"""
Unpruned Baseline

Returns the model as-is without any pruning.
Used as a reference for performance comparison.
"""

import torch.nn as nn

def unpruned_baseline(model: nn.Module, pruning_ratio: float = 0.0, **kwargs) -> nn.Module:
    """
    Returns the model without any changes.
    
    Args:
        model: PyTorch model
        pruning_ratio: Ignored
        **kwargs: Ignored
        
    Returns:
        Original model
    """
    print(f"Unpruned Baseline: Keeping model as-is (Pruning ratio: 0.0%)")
    return model
