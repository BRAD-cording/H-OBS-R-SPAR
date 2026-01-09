import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import copy
from .hobs import HOBSSensitivityAnalyzer
from .rspar import RSPARAgent, MDPState
from utils.pruner import PhysicalPruner
import numpy as np

class HOBSRSPARPruner:
    """
    H-OBS/R-SPAR 完整剪枝器类
    
    整合了 H-OBS 敏感度分析和 R-SPAR 预算分配的端到端剪枝框架
    
    核心流程：
    1. H-OBS 敏感度分析：计算各层的二阶敏感度分数
    2. R-SPAR 预算分配：使用 RL 代理学习最优的层级剪枝比例
    3. 物理剪枝：根据敏感度和预算分配执行结构化剪枝
    """
    
    def __init__(self, model: nn.Module, damping: float = 1e-3, use_kfac: bool = True,
                 rspar_hidden_dim: int = 256, rspar_lr: float = 3e-4):
        """
        初始化 H-OBS/R-SPAR 剪枝器
        
        Args:
            model: 要剪枝的 PyTorch 模型
            damping: H-OBS 中 Hessian 近似的阻尼系数
            use_kfac: 是否使用 K-FAC 近似
            rspar_hidden_dim: R-SPAR 代理网络的隐藏层维度
            rspar_lr: R-SPAR 代理的学习率
        """
        self.model = copy.deepcopy(model)
        
        # H-OBS 敏感度分析器
        self.hobs_analyzer = HOBSSensitivityAnalyzer(
            model, 
            damping=damping, 
            use_kfac=use_kfac
        )
        
        # R-SPAR 预算分配代理
        self.rspar_agent = None  # 会在计算敏感度后初始化
        self.rspar_hidden_dim = rspar_hidden_dim
        self.rspar_lr = rspar_lr
        
        # 物理剪枝器
        self.physical_pruner = None  # 会在执行剪枝时初始化
        
        # 敏感度分数
        self.sensitivity_scores = None
        
        # 确保模型在评估模式
        self.model.eval()
    
    def compute_sensitivity(self, dataloader: torch.utils.data.DataLoader, 
                           criterion: Optional[nn.Module] = None, 
                           num_batches: int = 10):
        """
        计算各层的敏感度分数
        
        Args:
            dataloader: 用于计算 Hessian 的数据集加载器
            criterion: 损失函数（默认使用交叉熵损失）
            num_batches: 用于计算的批次数量
        """
        print("\n[H-OBS/R-SPAR] 第 1 步: H-OBS 敏感度分析")
        self.sensitivity_scores = self.hobs_analyzer.compute_sensitivity(
            dataloader, 
            criterion=criterion, 
            num_batches=num_batches
        )
        
        # 初始化 R-SPAR 代理
        num_layers = len(self.sensitivity_scores)
        self.rspar_agent = RSPARAgent(
            num_layers=num_layers,
            hidden_dim=self.rspar_hidden_dim,
            lr=self.rspar_lr
        )
        
        print(f"✓ 敏感度分析完成，共分析了 {num_layers} 个可剪枝层")
    
    def allocate_budgets(self, num_episodes: int = 100, target_flops_reduction: float = 0.5):
        """
        使用 R-SPAR 代理分配剪枝预算
        
        Args:
            num_episodes: RL 训练的回合数
            target_flops_reduction: 目标 FLOPs 减少比例
            
        Returns:
            剪枝预算：层名称到剪枝比例的字典
        """
        if not self.sensitivity_scores:
            raise ValueError("请先调用 compute_sensitivity 方法计算敏感度分数")
        
        print("\n[H-OBS/R-SPAR] 第 2 步: R-SPAR 预算分配")
        print(f"训练 RL 代理 {num_episodes} 个回合...")
        
        # 简化的 RL 训练过程（实际应用中需要完整的环境模拟）
        num_layers = len(self.sensitivity_scores)
        
        # 训练 RL 代理（简化版）
        for episode in range(num_episodes):
            # 创建初始状态
            layer_features = []
            for scores in self.sensitivity_scores.values():
                # 每层特征：平均敏感度、敏感度标准差、通道数
                avg_sens = scores.mean().item()
                std_sens = scores.std().item()
                num_channels = scores.numel()
                layer_features.append([avg_sens, std_sens, num_channels])
            
            layer_features = np.array(layer_features)
            initial_ratios = np.random.uniform(0.1, 0.9, num_layers)
            
            state = MDPState(layer_features, initial_ratios, accuracy_drop=0.0)
            
            # 选择动作
            action, log_prob, value = self.rspar_agent.select_action(state)
            
            # 计算奖励（简化版：基于剪枝比例与目标的接近程度）
            total_ratio = action.mean().item()
            reward = -abs(total_ratio - target_flops_reduction) * 100  # 惩罚与目标的偏差
            
            # 存储经验
            self.rspar_agent.store_transition(state, action, reward, True, log_prob, value)
            
            # 每 10 个回合更新一次网络
            if (episode + 1) % 10 == 0:
                self.rspar_agent.update()
        
        # 使用训练好的代理分配预算
        budgets = self.rspar_agent.allocate_budgets(self.sensitivity_scores)
        
        print(f"✓ 预算分配完成")
        return budgets
    
    def prune(self, dataloader: torch.utils.data.DataLoader, 
             target_flops_reduction: float = 0.5, 
             num_batches_sensitivity: int = 10, 
             rspar_num_episodes: int = 100,
             example_input: Optional[torch.Tensor] = None) -> nn.Module:
        """
        执行完整的 H-OBS/R-SPAR 剪枝流程
        
        Args:
            dataloader: 用于计算敏感度的数据集加载器
            target_flops_reduction: 目标 FLOPs 减少比例
            num_batches_sensitivity: 用于敏感度分析的批次数量
            rspar_num_episodes: R-SPAR 训练的回合数
            example_input: 用于构建依赖图的示例输入
            
        Returns:
            剪枝后的模型
        """
        # 第 1 步：计算敏感度
        self.compute_sensitivity(dataloader, num_batches=num_batches_sensitivity)
        
        # 第 2 步：分配预算
        budgets = self.allocate_budgets(
            num_episodes=rspar_num_episodes,
            target_flops_reduction=target_flops_reduction
        )
        
        # 第 3 步：生成剪枝计划
        print("\n[H-OBS/R-SPAR] 第 3 步: 生成剪枝计划")
        pruning_plan = self._generate_pruning_plan(budgets)
        
        # 第 4 步：执行物理剪枝
        print("\n[H-OBS/R-SPAR] 第 4 步: 执行物理剪枝")
        self.physical_pruner = PhysicalPruner(self.model, example_input=example_input)
        pruned_model = self.physical_pruner.prune(pruning_plan)
        
        print("\n✓ H-OBS/R-SPAR 剪枝流程完成！")
        return pruned_model
    
    def _generate_pruning_plan(self, budgets: Dict[str, float]) -> Dict[str, List[int]]:
        """
        根据预算分配生成剪枝计划
        
        Args:
            budgets: 层名称到剪枝比例的字典
            
        Returns:
            剪枝计划：层名称到要剪枝的通道索引的字典
        """
        pruning_plan = {}
        
        for layer_name, prune_ratio in budgets.items():
            if layer_name not in self.sensitivity_scores:
                continue
                
            scores = self.sensitivity_scores[layer_name]
            num_channels = scores.numel()
            num_prune = int(num_channels * prune_ratio)
            
            if num_prune <= 0:
                continue
                
            # 选择敏感度最低的通道进行剪枝
            sorted_indices = torch.argsort(scores)
            prune_indices = sorted_indices[:num_prune].tolist()
            
            pruning_plan[layer_name] = prune_indices
        
        # 打印剪枝计划摘要
        total_pruned = sum(len(indices) for indices in pruning_plan.values())
        total_channels = sum(scores.numel() for scores in self.sensitivity_scores.values())
        
        print(f"剪枝计划摘要：")
        print(f"  总通道数: {total_channels}")
        print(f"  计划剪枝通道数: {total_pruned}")
        print(f"  实际剪枝比例: {total_pruned / total_channels:.1%}")
        
        return pruning_plan
    
    def get_sensitivity_scores(self) -> Dict[str, torch.Tensor]:
        """
        获取计算得到的敏感度分数
        
        Returns:
            敏感度分数字典
        """
        if not self.sensitivity_scores:
            raise ValueError("请先调用 compute_sensitivity 方法计算敏感度分数")
        
        return self.sensitivity_scores
    
    def validate(self, dataloader: torch.utils.data.DataLoader, 
                criterion: Optional[nn.Module] = None) -> Tuple[float, float]:
        """
        验证模型性能
        
        Args:
            dataloader: 验证数据集加载器
            criterion: 损失函数（默认使用交叉熵损失）
            
        Returns:
            验证准确率和平均损失
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        device = next(self.model.parameters()).device
        self.model.eval()
        
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                total_loss += loss.item() * targets.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / total
        
        return accuracy, avg_loss

