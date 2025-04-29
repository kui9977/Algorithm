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
    """加载模型"""
    model = MetalClassifier(input_dim, hidden_dims, num_classes)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
