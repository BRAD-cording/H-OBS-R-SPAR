import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class JTPPruner:
    """
    Joint Threshold Pruning (JTP) for Neural Networks
    联合阈值剪枝：同时优化所有层的剪枝阈值
    """
    
    def __init__(self, model, pruning_rate=0.5, learning_rate=1e-3, temperature=1.0):
        """
        初始化JTP剪枝器
        
        参数:
        - model: 要剪枝的PyTorch模型
        - pruning_rate: 整体剪枝比例（0-1之间）
        - learning_rate: 学习率
        - temperature: 温度参数，用于软化阈值选择
        """
        self.model = model
        self.pruning_rate = pruning_rate
        self.lr = learning_rate
        self.temperature = temperature
        
        # 保存原始权重
        self.original_weights = {}
        self.layer_names = []
        
        for name, param in model.named_parameters():
            if param.requires_grad and 'weight' in name:
                self.original_weights[name] = param.clone().detach()
                self.layer_names.append(name)
        
        # 初始化每个层的阈值参数
        self.thresholds = nn.ParameterList()
        for name in self.layer_names:
            # 初始阈值设为权重的均值
            mean_val = torch.mean(torch.abs(self.original_weights[name])).item()
            self.thresholds.append(nn.Parameter(torch.tensor(mean_val, dtype=torch.float32)))
    
    def prune(self, train_loader, val_loader, num_epochs=10):
        """
        执行联合阈值剪枝
        
        参数:
        - train_loader: 训练数据加载器
        - val_loader: 验证数据加载器
        - num_epochs: 总训练轮数
        """
        criterion = nn.CrossEntropyLoss()
        
        # 创建优化器，同时优化阈值和模型权重
        optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.thresholds),
            lr=self.lr
        )
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # 训练阶段
            self.model.train()
            total_train_loss = 0
            
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                
                # 应用剪枝
                self._apply_pruning()
                
                # 前向传播
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Average Train Loss: {avg_train_loss:.4f}")
            
            # 验证阶段
            val_acc = self._validate(val_loader, criterion)
            print(f"Validation Accuracy: {val_acc:.4f}")
        
        # 最终剪枝
        self._final_prune()
        return self.model
    
    def _apply_pruning(self):
        """
        应用剪枝到模型权重（使用软化的阈值函数）
        """
        for i, name in enumerate(self.layer_names):
            weight = self.original_weights[name]
            threshold = self.thresholds[i]
            
            # 使用软化的阈值函数（Gumbel-Softmax类似的连续近似）
            abs_weight = torch.abs(weight)
            mask = torch.sigmoid((abs_weight - threshold) / self.temperature)
            
            # 应用掩码
            pruned_weight = weight * mask
            
            # 更新模型权重
            for param_name, param in self.model.named_parameters():
                if param_name == name:
                    param.data = pruned_weight
                    break
    
    def _final_prune(self):
        """
        最终剪枝：使用硬阈值移除权重
        """
        for i, name in enumerate(self.layer_names):
            weight = self.original_weights[name]
            threshold = self.thresholds[i].item()
            
            # 创建硬掩码
            mask = torch.abs(weight) > threshold
            pruned_weight = weight * mask
            
            # 更新模型权重
            for param_name, param in self.model.named_parameters():
                if param_name == name:
                    param.data = pruned_weight
                    break
    
    def _validate(self, val_loader, criterion):
        """
        验证模型性能
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # 应用剪枝
                self._apply_pruning()
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return correct / total
    
    def get_pruning_statistics(self):
        """
        获取剪枝统计信息
        """
        stats = {}
        total_params = 0
        pruned_params = 0
        
        for i, name in enumerate(self.layer_names):
            weight = self.original_weights[name]
            threshold = self.thresholds[i].item()
            
            # 计算剪枝统计
            num_params = weight.numel()
            num_pruned = (torch.abs(weight) <= threshold).sum().item()
            
            stats[name] = {
                'total_params': num_params,
                'pruned_params': num_pruned,
                'pruning_rate': num_pruned / num_params,
                'threshold': threshold
            }
            
            total_params += num_params
            pruned_params += num_pruned
        
        stats['overall'] = {
            'total_params': total_params,
            'pruned_params': pruned_params,
            'pruning_rate': pruned_params / total_params
        }
        
        return stats
