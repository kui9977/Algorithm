import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class MetalClassifier(nn.Module):
    """金属材料分类模型"""
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.3):
        super(MetalClassifier, self).__init__()
        
        # 创建层列表
        layers = []
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 中间层
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 最后的分类层
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        
        # 创建序列模型
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)

class EnsembleModel:
    """集成模型"""
    def __init__(self, models):
        self.models = models
    
    def predict(self, x):
        """集成预测"""
        # 获取每个模型的预测
        predictions = [model(x).softmax(dim=1) for model in self.models]
        
        # 计算平均预测概率
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        return ensemble_pred

def save_model(model, path='D:\\Projects\\Python_projects\\dpl\\Algorithm\\Algorithm_material_system\\models\\metal_classifier.pth'):
    """保存模型"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(input_dim, hidden_dims, num_classes, path='D:\\Projects\\Python_projects\\dpl\\Algorithm\\Algorithm_material_system\\models\\metal_classifier.pth'):
    """加载模型
    
    参数:
    - input_dim: 输入特征维度
    - hidden_dims: 隐藏层维度列表
    - num_classes: 类别数量 (会根据模型权重检测到的类别数自动调整)
    - path: 模型权重路径
    """
    # 先检查模型权重中的实际类别数
    try:
        # 加载状态字典
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        
        # 检查最后一层权重尺寸，获取实际类别数
        if 'model.12.weight' in state_dict:
            actual_classes = state_dict['model.12.weight'].size(0)
            if actual_classes != num_classes:
                print(f"警告: 模型权重文件包含 {actual_classes} 个类别，与请求的 {num_classes} 个类别不匹配")
                print(f"将使用模型权重中的类别数 {actual_classes}")
                num_classes = actual_classes
    except Exception as e:
        print(f"读取模型权重信息出错，将使用提供的类别数 {num_classes}: {e}")

    # 创建具有正确类别数的模型
    model = MetalClassifier(input_dim, hidden_dims, num_classes)
    
    # 尝试加载权重
    try:
        # 尝试从状态字典加载模型
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        print(f"成功加载模型，类别数: {num_classes}")
    except Exception as e:
        print(f"标准加载模型失败: {e}")
        try:
            # 尝试非严格加载
            model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
            print("使用非严格模式加载模型（部分参数可能不匹配）")
        except Exception as e2:
            print(f"非严格加载也失败: {e2}")
            raise ValueError(f"无法加载模型: {e2}")
    
    # 设置为评估模式
    model.eval()
    return model
