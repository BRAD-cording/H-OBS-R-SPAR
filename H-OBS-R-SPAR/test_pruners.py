import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入所有剪枝器
from methods import get_pruner, AVAILABLE_PRUNERS

# 定义一个简单的CNN模型用于测试
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 8 * 8, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc(x)
        return x

# 准备测试数据
def prepare_test_data():
    # 创建随机数据
    inputs = torch.randn(100, 3, 16, 16)
    targets = torch.randint(0, 10, (100,))
    
    # 创建数据集和数据加载器
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    
    return inputs, dataloader

# 测试单个剪枝器
def test_pruner(pruner_name, model, inputs, dataloader, pruning_ratio=0.3):
    print(f"\n=== 测试 {pruner_name} 剪枝器 ===")
    
    try:
        # 初始化剪枝器
        pruner = get_pruner(pruner_name, model, pruning_ratio=pruning_ratio, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 测试剪枝前的模型性能
        print("剪枝前:")
        if hasattr(pruner, 'validate'):
            accuracy, loss = pruner.validate(dataloader, loss_fn=nn.CrossEntropyLoss())
            print(f"  准确率: {accuracy:.4f}, 损失: {loss:.4f}")
        
        # 执行剪枝
        if pruner_name == 'dep_graph':
            # DepGraph需要示例输入来构建依赖图
            pruner.prune(inputs=inputs)
        elif pruner_name in ['udfc'] and hasattr(pruner, 'prune'):
            # UDFC可能需要dataloader来计算激活
            pruner.prune(dataloader=dataloader)
        elif hasattr(pruner, 'prune'):
            # 其他剪枝器
            pruner.prune()
        else:
            print(f"  警告: {pruner_name} 没有prune方法")
            return False
        
        # 测试剪枝后的模型性能
        print("剪枝后:")
        if hasattr(pruner, 'validate'):
            accuracy, loss = pruner.validate(dataloader, loss_fn=nn.CrossEntropyLoss())
            print(f"  准确率: {accuracy:.4f}, 损失: {loss:.4f}")
        
        # 获取剪枝统计信息
        if hasattr(pruner, 'get_pruning_statistics'):
            stats = pruner.get_pruning_statistics()
            print(f"  剪枝统计: {stats}")
        
        print(f"  ✓ {pruner_name} 测试通过")
        return True
        
    except Exception as e:
        print(f"  ✗ {pruner_name} 测试失败: {str(e)}")
        return False

# 主测试函数
def main():
    print("=== 开始测试剪枝算法 ===")
    
    # 准备测试数据
    inputs, dataloader = prepare_test_data()
    
    # 测试结果统计
    results = {}
    
    # 测试所有剪枝器
    for pruner_name in AVAILABLE_PRUNERS.keys():
        # 创建新模型实例
        model = SimpleCNN()
        if torch.cuda.is_available():
            model = model.cuda()
        
        # 测试剪枝器
        success = test_pruner(pruner_name, model, inputs, dataloader)
        results[pruner_name] = success
    
    # 输出测试总结
    print("\n=== 测试总结 ===")
    passed = sum(results.values())
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    for pruner_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{pruner_name}: {status}")

if __name__ == "__main__":
    main()
