import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any

class UDFCPruner:
    """Uniform Distributed Feature Channels (UDFC) Pruner Implementation"""
    
    def __init__(self, model: nn.Module, pruning_ratio: float = 0.5, 
                 importance_metric: str = 'l1_norm', device: str = 'cuda'):
        """
        初始化UDFC剪枝器
        
        Args:
            model: 要剪枝的模型
            pruning_ratio: 剪枝比例 (0-1)
            importance_metric: 特征重要性度量 ('l1_norm', 'l2_norm', 'activation')
            device: 设备类型 ('cuda' 或 'cpu')
        """
        self.model = model.to(device)
        self.pruning_ratio = pruning_ratio
        self.importance_metric = importance_metric
        self.device = device
        self.pruned_layers = []
        self.importance_scores = {}
        
    def calculate_importance(self, dataloader: torch.utils.data.DataLoader = None):
        """
        计算特征通道的重要性分数
        
        Args:
            dataloader: 用于计算激活的数据集加载器（仅当 importance_metric='activation' 时需要）
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if self.importance_metric == 'l1_norm':
                    # L1范数作为重要性度量
                    importance = torch.sum(torch.abs(module.weight), dim=tuple(range(1, module.weight.dim())))
                elif self.importance_metric == 'l2_norm':
                    # L2范数作为重要性度量
                    importance = torch.sqrt(torch.sum(module.weight ** 2, dim=tuple(range(1, module.weight.dim()))))
                elif self.importance_metric == 'activation':
                    # 使用激活值作为重要性度量
                    importance = self._calculate_activation_importance(module, dataloader, name)
                else:
                    raise ValueError(f"不支持的重要性度量: {self.importance_metric}")
                
                self.importance_scores[name] = importance
        
    def _calculate_activation_importance(self, module: nn.Module, 
                                       dataloader: torch.utils.data.DataLoader, 
                                       layer_name: str):
        """
        使用激活值计算重要性
        
        Args:
            module: 要计算的模块
            dataloader: 数据集加载器
            layer_name: 层名称
            
        Returns:
            激活重要性分数
        """
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(output.detach())
        
        # 注册前向钩子
        hook = module.register_forward_hook(hook_fn)
        
        # 运行少量数据以收集激活
        self.model.eval()
        with torch.no_grad():
            for i, (data, _) in enumerate(dataloader):
                if i >= 10:  # 只使用前10个batch
                    break
                data = data.to(self.device)
                self.model(data)
        
        # 移除钩子
        hook.remove()
        
        if not activations:
            raise ValueError("未收集到激活值")
        
        # 计算激活的统计信息作为重要性
        all_activations = torch.cat(activations, dim=0)
        if isinstance(module, nn.Conv2d):
            # 对卷积层，计算每个通道的激活均值
            importance = torch.mean(torch.abs(all_activations), dim=(0, 2, 3))
        else:  # nn.Linear
            importance = torch.mean(torch.abs(all_activations), dim=0)
        
        return importance
    
    def _apply_uniform_distribution(self, importance: torch.Tensor):
        """
        应用均匀分布策略选择要剪枝的通道
        
        Args:
            importance: 通道重要性分数
            
        Returns:
            保留的通道索引
        """
        num_channels = importance.shape[0]
        num_to_keep = int(num_channels * (1 - self.pruning_ratio))
        
        if num_to_keep <= 0:
            raise ValueError("剪枝比例过高，没有通道保留")
        
        # 对重要性进行排序
        sorted_indices = torch.argsort(importance, descending=True)
        
        # 均匀选择要保留的通道
        keep_indices = sorted_indices[::max(1, len(sorted_indices) // num_to_keep)][:num_to_keep]
        
        # 确保索引是唯一且排序的
        keep_indices = torch.unique(keep_indices).sort().values
        
        # 如果选择的通道不足，补充最重要的通道
        if len(keep_indices) < num_to_keep:
            additional_indices = sorted_indices[~torch.isin(sorted_indices, keep_indices)][:num_to_keep - len(keep_indices)]
            keep_indices = torch.cat([keep_indices, additional_indices]).sort().values
        
        return keep_indices
    
    def prune(self, dataloader: torch.utils.data.DataLoader = None):
        """
        执行UDFC剪枝
        
        Args:
            dataloader: 用于计算激活的数据集加载器（仅当 importance_metric='activation' 时需要）
        """
        # 计算重要性
        self.calculate_importance(dataloader)
        
        # 对每个卷积层和全连接层进行剪枝
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name in self.importance_scores:
                importance = self.importance_scores[name]
                keep_indices = self._apply_uniform_distribution(importance)
                
                # 执行剪枝
                self._prune_layer(name, module, keep_indices)
    
    def _prune_layer(self, layer_name: str, module: nn.Module, keep_indices: torch.Tensor):
        """
        剪枝指定层
        
        Args:
            layer_name: 层名称
            module: 要剪枝的模块
            keep_indices: 要保留的通道索引
        """
        # 记录剪枝信息
        self.pruned_layers.append({
            'name': layer_name,
            'original_size': module.out_channels if isinstance(module, nn.Conv2d) else module.out_features,
            'pruned_size': len(keep_indices),
            'keep_indices': keep_indices.cpu().numpy()
        })
        
        # 更新权重
        if isinstance(module, nn.Conv2d):
            # 剪枝卷积层
            module.weight.data = module.weight.data[keep_indices]
            if module.bias is not None:
                module.bias.data = module.bias.data[keep_indices]
            # 更新输出通道数
            module.out_channels = len(keep_indices)
        else:  # nn.Linear
            # 剪枝全连接层
            module.weight.data = module.weight.data[keep_indices]
            if module.bias is not None:
                module.bias.data = module.bias.data[keep_indices]
            # 更新输出特征数
            module.out_features = len(keep_indices)
        
        # 更新下一层的输入通道数（如果是卷积层后面的层）
        self._update_next_layer_input(module, layer_name, keep_indices)
    
    def _update_next_layer_input(self, pruned_module: nn.Module, pruned_layer_name: str, keep_indices: torch.Tensor):
        """
        更新下一层的输入通道数以匹配当前层的输出通道数
        
        Args:
            pruned_module: 已剪枝的模块
            pruned_layer_name: 已剪枝的层名称
            keep_indices: 保留的通道索引
        """
        # 这里需要根据模型结构进行更复杂的处理
        # 简单起见，我们只处理连续的卷积层或全连接层
        layers = list(self.model.named_modules())
        pruned_index = -1
        
        for i, (name, module) in enumerate(layers):
            if name == pruned_layer_name:
                pruned_index = i
                break
        
        if pruned_index != -1 and pruned_index + 1 < len(layers):
            next_name, next_module = layers[pruned_index + 1]
            if isinstance(next_module, (nn.Conv2d, nn.Linear)):
                if isinstance(pruned_module, nn.Conv2d) and isinstance(next_module, nn.Conv2d):
                    # 卷积层 -> 卷积层：更新下一层的输入通道和权重
                    next_module.in_channels = len(keep_indices)
                    # 对于2D卷积，权重形状是 [out_channels, in_channels, kernel_size, kernel_size]
                    next_module.weight.data = next_module.weight.data[:, keep_indices]
                elif isinstance(pruned_module, nn.Linear) and isinstance(next_module, nn.Linear):
                    # 全连接层 -> 全连接层：更新下一层的输入特征和权重
                    next_module.in_features = len(keep_indices)
                    # 对于全连接层，权重形状是 [out_features, in_features]
                    next_module.weight.data = next_module.weight.data[:, keep_indices]
    
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
        total_pruned = sum(layer['pruned_size'] for layer in self.pruned_layers)
        total_pruned_channels = total_original - total_pruned
        
        return {
            'total_layers_pruned': len(self.pruned_layers),
            'total_original_channels': total_original,
            'total_pruned_channels': total_pruned_channels,
            'total_pruned_size': total_pruned,
            'pruning_ratio': total_pruned_channels / total_original if total_original > 0 else 0,
            'layer_details': self.pruned_layers
        }
    
    def get_pruned_model(self):
        """
        获取剪枝后的模型
        
        Returns:
            剪枝后的模型
        """
        return self.model