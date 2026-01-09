import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy

class MDPState:
    """
    MDP状态表示用于R-SPAR的预算分配
    """
    def __init__(self, layer_features: np.ndarray, current_ratios: np.ndarray, accuracy_drop: float):
        """
        初始化MDP状态
        
        Args:
            layer_features: 层特征向量（包含敏感度分数、通道数等）
            current_ratios: 当前层剪枝比例
            accuracy_drop: 当前精度下降
        """
        self.layer_features = layer_features
        self.current_ratios = current_ratios
        self.accuracy_drop = accuracy_drop
    
    def to_tensor(self) -> torch.Tensor:
        """
        将状态转换为张量
        
        Returns:
            状态张量
        """
        features = np.concatenate([
            self.layer_features.flatten(),
            self.current_ratios.flatten(),
            [self.accuracy_drop]
        ])
        return torch.tensor(features, dtype=torch.float32)

class Actor(nn.Module):
    """
    PPO算法的Actor网络
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        初始化Actor网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.std = nn.Parameter(torch.ones(action_dim) * 0.1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量
            
        Returns:
            动作均值和标准差
        """
        mean = torch.sigmoid(self.network(state))  # 确保输出在0-1之间
        std = self.std.exp().clamp(min=1e-3, max=1.0)  # 确保标准差为正且合理
        return mean, std
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        选择动作
        
        Args:
            state: 状态张量
            deterministic: 是否使用确定性策略
            
        Returns:
            动作张量
        """
        mean, std = self.forward(state)
        if deterministic:
            return mean
        
        # 采样动作
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        
        # 确保动作在0-1之间
        return torch.clamp(action, 0.0, 1.0)

class Critic(nn.Module):
    """
    PPO算法的Critic网络
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        """
        初始化Critic网络
        
        Args:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
        """
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态张量
            
        Returns:
            状态值函数
        """
        return self.network(state)

class RSPARAgent:
    """
    强化学习驱动的结构化剪枝代理（R-SPAR）
    
    使用PPO算法学习最优的层级剪枝比例分配策略
    """
    
    def __init__(self, num_layers: int, layer_feature_dim: int = 3, hidden_dim: int = 256,
                 lr: float = 3e-4, gamma: float = 0.99, clip_epsilon: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01):
        """
        初始化R-SPAR代理
        
        Args:
            num_layers: 可剪枝层的数量
            layer_feature_dim: 每层的特征维度
            hidden_dim: 网络隐藏层维度
            lr: 学习率
            gamma: 折扣因子
            clip_epsilon: PPO裁剪参数
            value_coef: 值函数损失系数
            entropy_coef: 熵正则化系数
        """
        self.num_layers = num_layers
        self.layer_feature_dim = layer_feature_dim
        
        # 状态维度：层特征 + 当前剪枝比例 + 精度下降
        state_dim = num_layers * layer_feature_dim + num_layers + 1
        action_dim = num_layers  # 每个层的剪枝比例
        
        # 创建Actor和Critic网络
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )
        
        # PPO参数
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def select_action(self, state: MDPState, deterministic: bool = False) -> Tuple[torch.Tensor, float, float]:
        """
        选择动作
        
        Args:
            state: MDP状态
            deterministic: 是否使用确定性策略
            
        Returns:
            动作、对数概率和状态值
        """
        state_tensor = state.to_tensor().unsqueeze(0)
        
        # 获取动作分布
        mean, std = self.actor(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        
        # 确保动作在0-1之间
        action = torch.clamp(action, 0.0, 1.0)
        
        # 计算对数概率和熵
        log_prob = dist.log_prob(action).sum(dim=1)
        
        # 获取状态值
        value = self.critic(state_tensor)
        
        return action.squeeze(), log_prob.item(), value.squeeze().item()
    
    def store_transition(self, state: MDPState, action: torch.Tensor, reward: float, 
                        done: bool, log_prob: float, value: float):
        """
        存储转换样本
        
        Args:
            state: MDP状态
            action: 动作
            reward: 奖励
            done: 是否结束
            log_prob: 对数概率
            value: 状态值
        """
        self.states.append(state.to_tensor())
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def update(self, num_epochs: int = 10, batch_size: int = 32):
        """
        更新Actor和Critic网络
        
        Args:
            num_epochs: 更新轮数
            batch_size: 批次大小
        """
        # 计算回报和优势函数
        returns = self._compute_returns()
        advantages = self._compute_advantages(returns)
        
        # 转换为张量
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # 归一化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 执行PPO更新
        for epoch in range(num_epochs):
            # 打乱数据
            indices = torch.randperm(states.size(0))
            
            for i in range(0, states.size(0), batch_size):
                # 获取批次数据
                batch_indices = indices[i:i+batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # 计算当前策略的对数概率
                mean, std = self.actor(batch_states)
                dist = torch.distributions.Normal(mean, std)
                batch_log_probs = dist.log_prob(batch_actions).sum(dim=1)
                
                # 计算概率比率
                ratio = torch.exp(batch_log_probs - batch_old_log_probs)
                
                # PPO损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 计算熵
                entropy = dist.entropy().sum(dim=1).mean()
                
                # Critic损失
                values = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(values, batch_returns)
                
                # 总损失
                total_loss = (
                    actor_loss + 
                    self.value_coef * critic_loss - 
                    self.entropy_coef * entropy
                )
                
                # 更新网络
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        
        # 清空经验缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def _compute_returns(self) -> List[float]:
        """
        计算回报
        
        Returns:
            回报列表
        """
        returns = []
        discounted_reward = 0
        
        for reward, done in reversed(list(zip(self.rewards, self.dones))):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        return returns
    
    def _compute_advantages(self, returns: List[float]) -> List[float]:
        """
        计算优势函数
        
        Args:
            returns: 回报列表
            
        Returns:
            优势函数列表
        """
        advantages = []
        for value, return_ in zip(self.values, returns):
            advantages.append(return_ - value)
        
        return advantages
    
    def allocate_budgets(self, sensitivity_scores: Dict[str, torch.Tensor], 
                        initial_ratios: Optional[List[float]] = None) -> Dict[str, float]:
        """
        分配剪枝预算（简化版本，用于实际剪枝）
        
        Args:
            sensitivity_scores: 各层的敏感度分数
            initial_ratios: 初始剪枝比例
            
        Returns:
            层名称到剪枝比例的字典
        """
        # 如果没有初始比例，使用均匀分布
        if initial_ratios is None:
            initial_ratios = [0.5] * self.num_layers
        
        # 创建初始状态
        layer_features = []
        for scores in sensitivity_scores.values():
            # 每层特征：平均敏感度、敏感度标准差、通道数
            avg_sens = scores.mean().item()
            std_sens = scores.std().item()
            num_channels = scores.numel()
            layer_features.append([avg_sens, std_sens, num_channels])
        
        layer_features = np.array(layer_features)
        current_ratios = np.array(initial_ratios)
        
        state = MDPState(layer_features, current_ratios, accuracy_drop=0.0)
        
        # 使用确定性策略选择动作
        with torch.no_grad():
            action, _, _ = self.select_action(state, deterministic=True)
        
        # 将动作转换为剪枝预算
        pruning_budgets = {}
        for i, (layer_name, _) in enumerate(sensitivity_scores.items()):
            pruning_budgets[layer_name] = action[i].item()
        
        return pruning_budgets

