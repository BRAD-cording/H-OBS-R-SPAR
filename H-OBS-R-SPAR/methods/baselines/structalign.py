import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class StructuralAlignmentPruner:
    """
    Structural Alignment Pruning for Neural Networks
    结构对齐剪枝：保持网络层之间的结构一致性
    """
    
    def __init__(self, model, pruning_rate=0.5, learning_rate=1e-3, alignment_weight=0.1):
        """
        初始化结构对齐剪枝器
        
        参数:
        - model: 要剪枝的PyTorch模型
        - pruning_rate: 剪枝比例（0-1之间）
        - learning_rate: 学习率
        - alignment_weight: 结构对齐损失的权重
        """
        self.model = model
        self.pruning_rate = pruning_rate
        self.lr = learning_rate
        self.alignment_weight = alignment_weight
        
        # 保存原始权重
        self.original_weights = {}
        self.layer_info = {}
        
        # 收集卷积层和线性层信息
        self.conv_layers = []
        self.linear_layers = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weight = module.weight.data.clone().detach()
                self.original_weights[name] = weight
                
                if isinstance(module, nn.Conv2d):
                    self.conv_layers.append(name)
                else:
                    self.linear_layers.append(name)
                
                # 保存层信息
                self.layer_info[name] = {
                    'module': module,
                    'type': 'conv' if isinstance(module, nn.Conv2d) else 'linear'
                }
        
        # 初始化神经元重要性得分
        self.neuron_importance = {}
        for name in self.original_weights.keys():
            weight = self.original_weights[name]
            if self.layer_info[name]['type'] == 'conv':
                # 卷积层：每个输出通道的重要性
                self.neuron_importance[name] = nn.Parameter(torch.ones(weight.size(0), dtype=torch.float32))
            else:
                # 线性层：每个输出神经元的重要性
                self.neuron_importance[name] = nn.Parameter(torch.ones(weight.size(0), dtype=torch.float32))
    
    def prune(self, train_loader, val_loader, num_epochs=10):
        """
        执行结构对齐剪枝
        
        参数:
        - train_loader: 训练数据加载器
        - val_loader: 验证数据加载器
        - num_epochs: 总训练轮数
        """
        criterion = nn.CrossEntropyLoss()
        
        # 创建优化器
        optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.neuron_importance.values()),
            lr=self.lr
        )
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            self.model.train()
            total_loss = 0
            
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                
                # 应用结构对齐的剪枝
                self._apply_structural_alignment()
                
                # 前向传播
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # 添加结构对齐损失
                alignment_loss = self._compute_alignment_loss()
                total_loss = loss + self.alignment_weight * alignment_loss
                
                total_loss.backward()
                optimizer.step()
            
            # 验证当前模型
            val_acc = self._validate(val_loader, criterion)
            print(f"Validation Accuracy: {val_acc:.4f}, Alignment Loss: {alignment_loss.item():.4f}")
        
        # 根据重要性得分执行最终剪枝
        self._final_prune()
        return self.model
    
    def _compute_neuron_importance(self):
        """
        计算神经元重要性得分
        """
        for name, weight in self.original_weights.items():
            if self.layer_info[name]['type'] == 'conv':
                # 卷积层：使用L1范数作为初始重要性
                self.neuron_importance[name] = nn.Parameter(
                    torch.sum(torch.abs(weight), dim=(1, 2, 3))
                )
            else:
                # 线性层：使用L1范数作为初始重要性
                self.neuron_importance[name] = nn.Parameter(
                    torch.sum(torch.abs(weight), dim=1)
                )
    
    def _compute_alignment_loss(self):
        """
        计算结构对齐损失
        目标是使相邻层的神经元重要性分布尽可能相似
        """
        alignment_loss = 0.0
        
        # 处理卷积层序列
        for i in range(len(self.conv_layers) - 1):
            layer1_name = self.conv_layers[i]
            layer2_name = self.conv_layers[i + 1]
            
            # 获取两层的重要性得分
            imp1 = self.neuron_importance[layer1_name]
            imp2 = self.neuron_importance[layer2_name]
            
            # 计算分布差异（KL散度）
            p = torch.softmax(imp1, dim=0)
            q = torch.softmax(imp2, dim=0)
            
            # 确保概率分布有效
            p = p + 1e-10
            q = q + 1e-10
            
            kl_div = torch.sum(p * torch.log(p / q))
            alignment_loss += kl_div
        
        # 处理线性层序列
        for i in range(len(self.linear_layers) - 1):
            layer1_name = self.linear_layers[i]
            layer2_name = self.linear_layers[i + 1]
            
            # 获取两层的重要性得分
            imp1 = self.neuron_importance[layer1_name]
            imp2 = self.neuron_importance[layer2_name]
            
            # 计算分布差异（KL散度）
            p = torch.softmax(imp1, dim=0)
            q = torch.softmax(imp2, dim=0)
            
            # 确保概率分布有效
            p = p + 1e-10
            q = q + 1e-10
            
            kl_div = torch.sum(p * torch.log(p / q))
            alignment_loss += kl_div
        
        return alignment_loss
    
    def _apply_structural_alignment(self):
        """
        应用结构对齐的剪枝掩码
        """
        for name in self.original_weights.keys():
            weight = self.original_weights[name]
            importance = self.neuron_importance[name]
            
            if self.layer_info[name]['type'] == 'conv':
                # 卷积层：为每个输出通道应用重要性得分
                mask = importance.view(-1, 1, 1, 1)
                pruned_weight = weight * mask
            else:
                # 线性层：为每个输出神经元应用重要性得分
                mask = importance.view(-1, 1)
                pruned_weight = weight * mask
            
            # 更新模型权重
            self.layer_info[name]['module'].weight.data = pruned_weight
    
    def _final_prune(self):
        """
        根据重要性得分执行最终剪枝
        """
        for name in self.original_weights.keys():
            importance = self.neuron_importance[name].detach()
            
            # 计算要保留的神经元数量
            num_neurons = importance.size(0)
            num_keep = int(num_neurons * (1 - self.pruning_rate))
            
            # 选择重要性最高的神经元
            _, keep_indices = torch.topk(importance, num_keep)
            keep_indices = torch.sort(keep_indices)[0]
            
            # 创建掩码
            mask = torch.zeros_like(importance)
            mask[keep_indices] = 1
            
            # 应用掩码
            weight = self.original_weights[name]
            if self.layer_info[name]['type'] == 'conv':
                pruned_weight = weight[keep_indices, :, :, :]
                # 更新卷积层
                self.layer_info[name]['module'].out_channels = num_keep
                self.layer_info[name]['module'].weight.data = pruned_weight
            else:
                pruned_weight = weight[keep_indices, :]
                # 更新线性层
                self.layer_info[name]['module'].out_features = num_keep
                self.layer_info[name]['module'].weight.data = pruned_weight
    
    def _validate(self, val_loader, criterion):
        """
        验证模型性能
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # 应用结构对齐
                self._apply_structural_alignment()
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return correct / total
