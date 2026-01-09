# 统一剪枝方法接口

from .hobs import HOBSSensitivityAnalyzer
from .rspar import RSPARAgent
from .hobs_rspar_pruner import HOBSRSPARPruner

# Import baseline pruning methods
from .baselines.depgraph import DepGraphPruner
from .baselines.jtp import JTPPruner
from .baselines.bilevel import BiLevelPruner
from .baselines.structalign import StructuralAlignmentPruner
from .baselines.udfc import UDFCPruner
from .baselines.unpruned import UnprunedPruner

# 可用的剪枝方法列表
AVAILABLE_PRUNERS = {
    # Main methods
    'hobs': HOBSSensitivityAnalyzer,
    'rsp': RSPARAgent,
    'hobs_rspar': HOBSRSPARPruner,
    
    # Baseline methods
    'unpruned': UnprunedPruner,
    'depgraph': DepGraphPruner,
    'jtp': JTPPruner,
    'bi_level': BiLevelPruner,
    'struct_align': StructuralAlignmentPruner,
    'udfc': UDFCPruner
}

def get_pruner(pruner_name: str, model, **kwargs):
    """
    根据名称获取相应的剪枝器实例
    
    Args:
        pruner_name: 剪枝器名称 ('bi_level', 'dep_graph', 'jtp', 'structural_alignment', 'udfc')
        model: 要剪枝的模型
        **kwargs: 剪枝器的初始化参数
        
    Returns:
        剪枝器实例
    """
    pruner_name = pruner_name.lower()
    if pruner_name not in AVAILABLE_PRUNERS:
        raise ValueError(f"不支持的剪枝器名称: {pruner_name}. 可用的剪枝器: {list(AVAILABLE_PRUNERS.keys())}")
    
    return AVAILABLE_PRUNERS[pruner_name](model, **kwargs)

# 导出所有剪枝器和工厂函数
__all__ = [
    # Main methods
    'HOBSSensitivityAnalyzer',
    'RSPARAgent',
    'HOBSRSPARPruner',
    
    # Baseline methods
    'UnprunedPruner',
    'DepGraphPruner',
    'JTPPruner',
    'BiLevelPruner',
    'StructuralAlignmentPruner',
    'UDFCPruner',
    
    # Utilities
    'AVAILABLE_PRUNERS',
    'get_pruner'
]
