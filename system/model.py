import torch
import torch.nn as nn
import torch.nn.functional as F

class MetalClassifier(nn.Module):
    """
    用于识别金属材料的多模态模型
    """
    
    def __init__(self, input_size, num_classes, hidden_size=128, dropout_rate=0.3):
        """
        初始化模型
        
        参数:
            input_size: 输入特征维度
            num_classes: 材料类别数量
            hidden_size: 隐藏层大小
            dropout_rate: Dropout比率，用于减轻过拟合
        """
        super(MetalClassifier, self).__init__()
        
        # 第一层全连接网络
        self.fc1 = nn.Linear(input_size, hidden_size*2)
        self.bn1 = nn.BatchNorm1d(hidden_size*2)  # 批归一化提高训练稳定性
        
        # 第二层全连接网络
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        # 第三层全连接网络
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.bn3 = nn.BatchNorm1d(hidden_size//2)
        
        # 输出层
        self.fc4 = nn.Linear(hidden_size//2, num_classes)
        
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征
            
        返回:
            分类的logits
        """
        # 第一层
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第二层
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 第三层
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # 输出层
        x = self.fc4(x)
        
        return x

class EnsembleModel(nn.Module):
    """
    集成多个模型进行预测
    """
    def __init__(self, input_size, num_classes, num_models=3, hidden_size=128, dropout_rate=0.3):
        """
        初始化集成模型
        
        参数:
            input_size: 输入特征维度
            num_classes: 材料类别数量
            num_models: 集成的模型数量
            hidden_size: 隐藏层大小
            dropout_rate: Dropout比率
        """
        super(EnsembleModel, self).__init__()
        
        # 创建多个基础模型
        self.models = nn.ModuleList([
            MetalClassifier(input_size, num_classes, hidden_size, dropout_rate)
            for _ in range(num_models)
        ])
    
    def forward(self, x):
        """
        前向传播，集成多个模型的输出
        
        参数:
            x: 输入特征
            
        返回:
            平均后的预测结果
        """
        # 获取每个模型的输出
        outputs = [model(x) for model in self.models]
        
        # 对输出取平均
        return torch.mean(torch.stack(outputs), dim=0)

if __name__ == "__main__":
    # 测试模型
    input_size = 20  # 假设特征数量为20
    num_classes = 104  # 材料类别数量
    
    # 测试单个模型
    model = MetalClassifier(input_size, num_classes)
    test_input = torch.randn(8, input_size)  # 批量大小为8的测试输入
    output = model(test_input)
    print(f"单个模型输出形状: {output.shape}")
    
    # 测试集成模型
    ensemble = EnsembleModel(input_size, num_classes)
    ensemble_output = ensemble(test_input)
    print(f"集成模型输出形状: {ensemble_output.shape}")
