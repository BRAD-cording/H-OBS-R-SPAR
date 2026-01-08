"""
Baseline Pruning Methods

This module contains implementations of state-of-the-art pruning methods
for comparison with H-OBS/R-SPAR.

Available Methods:
- DepGraph (CVPR 2023): Dependency-aware structural pruning
- JTP (CVPR 2024): Joint Training of Pruning and Fine-tuning
- Bi-Level (CVPR 2024): Bi-Level Structural Pruning
- StructAlign (ICCV 2023): Structure-Aligned Regularization
- UDFC (ICCV 2023): Unified Data-Free Compression

Usage:
    from models.baselines import dep_graph_prune, jtp_prune
    
    pruned_model = dep_graph_prune(model, pruning_ratio=0.5)
"""

from .depgraph import dep_graph_prune, DepGraphPruner
from .jtp import jtp_prune, JTPPruner
from .bilevel import bilevel_prune, BiLevelPruner
from .structalign import structalign_prune, StructAlignPruner
from .udfc import udfc_prune, UDFCPruner

from .unpruned import unpruned_baseline

__all__ = [
    # DepGraph
    'dep_graph_prune',
    'DepGraphPruner',
    
    # JTP
    'jtp_prune',
    'JTPPruner',
    
    # Bi-Level
    'bilevel_prune',
    'BiLevelPruner',
    
    # StructAlign
    'structalign_prune',
    'StructAlignPruner',
    
    # UDFC
    'udfc_prune',
    'UDFCPruner',

    # Unpruned
    'unpruned_baseline',
]

# Method registry for easy access
PRUNING_METHODS = {
    'depgraph': dep_graph_prune,
    'jtp': jtp_prune,
    'bilevel': bilevel_prune,
    'structalign': structalign_prune,
    'udfc': udfc_prune,
    'unpruned': unpruned_baseline,
}


def get_pruning_method(method_name: str):
    """
    Get pruning method by name.
    
    Args:
        method_name: Name of the pruning method
    
    Returns:
        Pruning function
    
    Example:
        >>> prune_fn = get_pruning_method('jtp')
        >>> pruned_model = prune_fn(model, dataloader, pruning_ratio=0.5)
    """
    method = PRUNING_METHODS.get(method_name.lower())
    if method is None:
        available = ', '.join(PRUNING_METHODS.keys())
        raise ValueError(
            f"Unknown pruning method: {method_name}. "
            f"Available methods: {available}"
        )
    return method
