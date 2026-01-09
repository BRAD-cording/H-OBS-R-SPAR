import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BiLevelPruner:
    """
    Bi-level Optimization for Efficient Neural Network Pruning
    双层优化框架：上层优化剪枝掩码，下层优化权重
    """
    
    def __init__(self, model, pruning_rate=0.5, learning_rate=1e-3, lambda_reg=1e-4):
        """
        初始化Bi-level剪枝器
        
        参数:
        - model: 要剪枝的PyTorch模型
        - pruning_rate: 剪枝比例（0-1之间）
        - learning_rate: 学习率
        - lambda_reg: 正则化参数
        """
        self.model = model
        self.pruning_rate = pruning_rate
        self.lr = learning_rate
        self.lambda_reg = lambda_reg
        
        # 保存原始权重
        self.original_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad and 'weight' in name:
                self.original_weights[name] = param.clone().detach()
        
        # 初始化掩码
        self.masks = {}
        self._init_masks()
    
    def _init_masks(self):
        """
        初始化剪枝掩码
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'weight' in name:
                # 初始化为全1掩码
                self.masks[name] = torch.ones_like(param)
    
    def prune(self, train_loader, val_loader, num_epochs=10, upper_epochs=5):
        """
        执行双层剪枝
        
        参数:
        - train_loader: 训练数据加载器
        - val_loader: 验证数据加载器
        - num_epochs: 总训练轮数
        - upper_epochs: 上层优化轮数
        """
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # 上层优化：更新掩码
            self._upper_level_optimization(val_loader, criterion, upper_epochs)
            
            # 下层优化：更新权重
            self._lower_level_optimization(train_loader, criterion)
            
            # 验证当前模型
            val_acc = self._validate(val_loader, criterion)
            print(f"Validation Accuracy: {val_acc:.4f}")
        
        return self.model
    
    def _upper_level_optimization(self, val_loader, criterion, num_epochs):
        """
        上层优化：优化剪枝掩码
        """
        # 创建掩码优化器
        mask_params = []
        for name, mask in self.masks.items():
            mask = mask.requires_grad_(True)
            mask_params.append(mask)
        
        optimizer = optim.Adam(mask_params, lr=self.lr)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for inputs, targets in val_loader:
                optimizer.zero_grad()
                
                # 应用掩码
                self._apply_masks()
                
                # 前向传播
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # 添加正则化项（L1正则化掩码）
                reg_loss = 0
                for mask in self.masks.values():
                    reg_loss += torch.norm(mask, p=1)
                
                total_loss = loss + self.lambda_reg * reg_loss
                total_loss.backward()
                optimizer.step()
                
                # 确保掩码值在[0, 1]范围内
                for name, mask in self.masks.items():
                    self.masks[name] = torch.clamp(mask, 0, 1)
            
            # 根据pruning_rate阈值化掩码
            self._threshold_masks()
    
    def _lower_level_optimization(self, train_loader, criterion):
        """
        下层优化：优化权重
        """
        # 创建权重优化器
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        
        self.model.train()
        total_loss = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            # 应用掩码
            self._apply_masks()
            
            # 前向传播
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    def _apply_masks(self):
        """
        应用掩码到模型权重
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'weight' in name and name in self.masks:
                param.data = self.original_weights[name] * self.masks[name]
    
    def _threshold_masks(self):
        """
        根据剪枝比例阈值化掩码
        """
        all_mask_values = []
        for name, mask in self.masks.items():
            all_mask_values.append(mask.view(-1))
        
        all_mask_values = torch.cat(all_mask_values)
        threshold = torch.quantile(all_mask_values, self.pruning_rate)
        
        # 应用阈值
        for name, mask in self.masks.items():
            self.masks[name] = (mask > threshold).float()
    
    def _validate(self, val_loader, criterion):
        """
        验证模型性能
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # 应用掩码
                self._apply_masks()
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return correct / total
    
    def get_pruned_model(self):
        """
        获取剪枝后的模型（移除零权重）
        """
        # 应用掩码
        self._apply_masks()
        
        # 创建新模型
        pruned_model = self.model.__class__()
        
        # 复制非零权重
        for name, param in self.model.named_parameters():
            if name in self.masks:
                mask = self.masks[name]
                pruned_param = param * mask
                setattr(pruned_model, name.split('.')[-1], pruned_param)
        
        return pruned_model
