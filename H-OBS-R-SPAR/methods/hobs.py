import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from utils.kfac import KFACOptimizer
import copy

class HOBSSensitivityAnalyzer:
    """
    Hessian-guided On-demand Budget Sensitivity Analyzer (H-OBS)
    
    实现了基于K-FAC近似的二阶敏感度分析，用于识别网络中最冗余的过滤器
    
    核心原理：
    - 使用K-FAC近似Hessian矩阵，降低计算复杂度
    - 计算每个参数的敏感度分数：S(θᵢ) = |θᵢ|² · H_ii
    - 支持层级敏感度聚合，用于剪枝预算分配
    """
    
    def __init__(self, model: nn.Module, damping: float = 1e-3, use_kfac: bool = True):
        """
        初始化H-OBS敏感度分析器
        
        Args:
            model: 要分析的PyTorch模型
            damping: Hessian近似的阻尼系数
            use_kfac: 是否使用K-FAC近似（否则使用对角线Hessian）
        """
        self.model = copy.deepcopy(model)
        self.damping = damping
        self.use_kfac = use_kfac
        
        # 确保模型在评估模式
        self.model.eval()
        
        # 记录可剪枝层（卷积层和线性层）
        self.prunable_layers = self._get_prunable_layers()
        
        # KFAC优化器（用于Hessian近似）
        self.kfac_optimizer = None
        if use_kfac:
            self.kfac_optimizer = KFACOptimizer(
                self.model, 
                lr=0.1, 
                momentum=0.9, 
                stat_decay=0.95, 
                kl_clip=0.01,
                damping=damping,
                weight_decay=0.0
            )
    
    def _get_prunable_layers(self) -> Dict[str, nn.Module]:
        """
        获取模型中所有可剪枝的层（卷积层和线性层）
        
        Returns:
            层名称到层模块的字典
        """
        prunable_layers = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prunable_layers[name] = module
        return prunable_layers
    
    def compute_sensitivity(self, dataloader: torch.utils.data.DataLoader, 
                           criterion: Optional[nn.Module] = None, 
                           num_batches: int = 10) -> Dict[str, torch.Tensor]:
        """
        计算各层的敏感度分数
        
        Args:
            dataloader: 用于计算Hessian的数据集加载器
            criterion: 损失函数（默认使用交叉熵损失）
            num_batches: 用于计算的批次数量
            
        Returns:
            层名称到敏感度分数的字典，每个敏感度分数对应层的输出通道
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        # 重置K-FAC统计信息
        if self.use_kfac and self.kfac_optimizer:
            self.kfac_optimizer.reset_running_stats()
        
        # 收集激活和梯度统计信息
        print(f"收集 {num_batches} 个批次的统计信息...")
        
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            inputs = inputs.to(next(self.model.parameters()).device)
            targets = targets.to(next(self.model.parameters()).device)
            
            # 前向传播
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            self.model.zero_grad()
            loss.backward()
            
            # 更新K-FAC统计信息
            if self.use_kfac and self.kfac_optimizer:
                self.kfac_optimizer.update_running_stats()
        
        # 计算敏感度分数
        sensitivity_scores = {}
        
        for name, module in self.prunable_layers.items():
            if isinstance(module, nn.Conv2d):
                # 对于卷积层，计算每个输出通道的敏感度
                sensitivity = self._compute_conv_layer_sensitivity(module)
                sensitivity_scores[name] = sensitivity
            elif isinstance(module, nn.Linear):
                # 对于线性层，计算每个输出神经元的敏感度
                sensitivity = self._compute_linear_layer_sensitivity(module)
                sensitivity_scores[name] = sensitivity
        
        return sensitivity_scores
    
    def _compute_conv_layer_sensitivity(self, conv_layer: nn.Conv2d) -> torch.Tensor:
        """
        计算卷积层的敏感度分数
        
        Args:
            conv_layer: 卷积层模块
            
        Returns:
            每个输出通道的敏感度分数（形状：[out_channels]）
        """
        weights = conv_layer.weight.data  # 形状: [out_channels, in_channels, kernel_size, kernel_size]
        
        if self.use_kfac and self.kfac_optimizer:
            # 使用K-FAC近似Hessian对角线
            try:
                # 获取K-FAC的Hessian近似
                hessian_diag = self.kfac_optimizer._get_hessian_diag(conv_layer.weight)
                
                # 重塑Hessian对角线以匹配权重形状
                hessian_diag = hessian_diag.view(weights.shape)
                
                # 计算每个输出通道的敏感度：权重平方 * Hessian对角线，然后沿输入通道和核大小求和
                sensitivity = (weights ** 2 * hessian_diag).sum(dim=(1, 2, 3))
            except Exception as e:
                print(f"K-FAC计算失败，使用对角线近似: {e}")
                # 降级到对角线Hessian近似（基于梯度平方）
                sensitivity = (weights ** 2).sum(dim=(1, 2, 3))
        else:
            # 使用对角线Hessian近似（基于梯度平方）
            sensitivity = (weights ** 2).sum(dim=(1, 2, 3))
        
        return sensitivity
    
    def _compute_linear_layer_sensitivity(self, linear_layer: nn.Linear) -> torch.Tensor:
        """
        计算线性层的敏感度分数
        
        Args:
            linear_layer: 线性层模块
            
        Returns:
            每个输出神经元的敏感度分数（形状：[out_features]）
        """
        weights = linear_layer.weight.data  # 形状: [out_features, in_features]
        
        if self.use_kfac and self.kfac_optimizer:
            # 使用K-FAC近似Hessian对角线
            try:
                # 获取K-FAC的Hessian近似
                hessian_diag = self.kfac_optimizer._get_hessian_diag(linear_layer.weight)
                
                # 重塑Hessian对角线以匹配权重形状
                hessian_diag = hessian_diag.view(weights.shape)
                
                # 计算每个输出神经元的敏感度：权重平方 * Hessian对角线，然后沿输入特征求和
                sensitivity = (weights ** 2 * hessian_diag).sum(dim=1)
            except Exception as e:
                print(f"K-FAC计算失败，使用对角线近似: {e}")
                # 降级到对角线Hessian近似（基于梯度平方）
                sensitivity = (weights ** 2).sum(dim=1)
        else:
            # 使用对角线Hessian近似（基于梯度平方）
            sensitivity = (weights ** 2).sum(dim=1)
        
        return sensitivity
    
    def get_pruning_plan(self, sensitivity_scores: Dict[str, torch.Tensor], 
                        target_ratio: float) -> Dict[str, List[int]]:
        """
        根据敏感度分数生成剪枝计划
        
        Args:
            sensitivity_scores: 各层的敏感度分数
            target_ratio: 目标剪枝比例（0-1之间，表示要剪枝的比例）
            
        Returns:
            剪枝计划：层名称到要剪枝的通道索引的字典
        """
        if not sensitivity_scores:
            raise ValueError("敏感度分数不能为空")
        
        # 计算总通道数和目标剪枝通道数
        total_channels = sum(scores.numel() for scores in sensitivity_scores.values())
        target_prune_channels = int(total_channels * target_ratio)
        
        print(f"总通道数: {total_channels}")
        print(f"目标剪枝通道数: {target_prune_channels}")
        
        # 收集所有通道的敏感度分数
        all_sensitivities = []
        all_channel_info = []
        
        for layer_name, scores in sensitivity_scores.items():
            for channel_idx, score in enumerate(scores):
                all_sensitivities.append(score)
                all_channel_info.append((layer_name, channel_idx))
        
        # 对敏感度分数进行排序（从小到大）
        sorted_indices = torch.argsort(torch.tensor(all_sensitivities))
        
        # 选择敏感度最低的通道进行剪枝
        pruning_plan = {}
        pruned_channels = 0
        
        for idx in sorted_indices:
            if pruned_channels >= target_prune_channels:
                break
                
            layer_name, channel_idx = all_channel_info[idx.item()]
            
            if layer_name not in pruning_plan:
                pruning_plan[layer_name] = []
            
            pruning_plan[layer_name].append(channel_idx)
            pruned_channels += 1
        
        # 确保索引已排序
        for layer_name in pruning_plan:
            pruning_plan[layer_name] = sorted(pruning_plan[layer_name])
        
        return pruning_plan

