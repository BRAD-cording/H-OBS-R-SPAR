import torch
import torch.nn as nn
from ..DepGraph.torch_pruning.dependency import DependencyGraph
from ..DepGraph.torch_pruning import ops

class DepGraphPruner:
    """Dependency Graph (DepGraph) Pruner Implementation"""
    
    def __init__(self, model: nn.Module, pruning_ratio: float = 0.5, 
                 importance_metric: str = 'l1_norm', device: str = 'cuda'):
        """
        初始化DepGraph剪枝器
        
        Args:
            model: 要剪枝的模型
            pruning_ratio: 剪枝比例 (0-1)
            importance_metric: 特征重要性度量 ('l1_norm', 'l2_norm')
            device: 设备类型 ('cuda' 或 'cpu')
        """
        self.model = model.to(device)
        self.pruning_ratio = pruning_ratio
        self.importance_metric = importance_metric
        self.device = device
        self.pruned_layers = []
        self.dep_graph = None
        
    def _build_dependency_graph(self, inputs: torch.Tensor):
        """
        构建依赖图
        
        Args:
            inputs: 模型输入的示例张量
        """
        self.dep_graph = DependencyGraph().build_dependency_graph(
            self.model, inputs=inputs
        )
    
    def calculate_importance(self, inputs: torch.Tensor = None):
        """
        计算特征通道的重要性分数
        
        Args:
            inputs: 模型输入的示例张量（用于构建依赖图）
        """
        if self.dep_graph is None and inputs is not None:
            self._build_dependency_graph(inputs)
        
        self.importance_scores = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if self.importance_metric == 'l1_norm':
                    # L1范数作为重要性度量
                    importance = torch.sum(torch.abs(module.weight), dim=tuple(range(1, module.weight.dim())))
                elif self.importance_metric == 'l2_norm':
                    # L2范数作为重要性度量
                    importance = torch.sqrt(torch.sum(module.weight ** 2, dim=tuple(range(1, module.weight.dim()))))
                else:
                    raise ValueError(f"不支持的重要性度量: {self.importance_metric}")
                
                self.importance_scores[name] = importance
    
    def prune(self, inputs: torch.Tensor, dataloader: torch.utils.data.DataLoader = None):
        """
        执行DepGraph剪枝
        
        Args:
            inputs: 模型输入的示例张量
            dataloader: 用于计算重要性的数据集加载器（可选）
        """
        # 构建依赖图
        self._build_dependency_graph(inputs)
        
        # 计算重要性
        self.calculate_importance(inputs)
        
        # 对每个卷积层进行剪枝
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                importance = self.importance_scores[name]
                
                # 计算要剪枝的通道数
                num_channels = importance.shape[0]
                num_to_prune = int(num_channels * self.pruning_ratio)
                
                if num_to_prune <= 0:
                    continue
                
                # 选择要剪枝的通道（重要性最低的通道）
                _, indices_to_prune = torch.topk(importance, num_to_prune, largest=False)
                indices_to_prune = sorted(indices_to_prune.cpu().numpy())
                
                # 获取剪枝组
                if isinstance(module, nn.Conv2d):
                    # 对卷积层，剪枝输出通道
                    pruning_fn = self.dep_graph.get_pruning_fn(module, 'prune_out_channels')
                else:
                    # 对全连接层，剪枝输出特征
                    pruning_fn = self.dep_graph.get_pruning_fn(module, 'prune_out_channels')
                
                # 执行剪枝
                if pruning_fn is not None:
                    group = self.dep_graph.get_pruning_group(module, pruning_fn, indices_to_prune)
                    group.prune()
                    
                    # 记录剪枝信息
                    self.pruned_layers.append({
                        'name': name,
                        'original_size': num_channels,
                        'pruned_size': num_channels - num_to_prune,
                        'indices_to_prune': indices_to_prune
                    })
    
    def validate(self, dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module = None):
        """
        验证剪枝后模型的性能
        
        Args:
            dataloader: 验证数据集加载器
            loss_fn: 损失函数
            
        Returns:
            准确率和损失值
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                if loss_fn is not None:
                    loss = loss_fn(output, target)
                    total_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader) if loss_fn is not None else None
        
        return accuracy, avg_loss
    
    def get_pruning_statistics(self):
        """
        获取剪枝统计信息
        
        Returns:
            剪枝统计字典
        """
        if not self.pruned_layers:
            return {}
        
        total_original = sum(layer['original_size'] for layer in self.pruned_layers)
        total_pruned = sum(len(layer['indices_to_prune']) for layer in self.pruned_layers)
        total_remaining = total_original - total_pruned
        
        return {
            'total_layers_pruned': len(self.pruned_layers),
            'total_original_channels': total_original,
            'total_pruned_channels': total_pruned,
            'total_remaining_channels': total_remaining,
            'pruning_ratio': total_pruned / total_original if total_original > 0 else 0,
            'layer_details': self.pruned_layers
        }
    
    def get_pruned_model(self):
        """
        获取剪枝后的模型
        
        Returns:
            剪枝后的模型
        """
        return self.model
