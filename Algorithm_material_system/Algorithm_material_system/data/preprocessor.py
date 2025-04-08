import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


class MaterialDataPreprocessor:
    def __init__(self):
        """
        初始化材料数据预处理器
        创建颜色到索引的映射字典、索引到颜色的映射字典和标准化器
        """
        self.color_to_idx = {}  # 颜色文本到索引的映射
        self.idx_to_color = {}  # 索引到颜色文本的映射
        self.numerical_scaler = StandardScaler()  # 用于标准化数值特征
        # 设置一个默认颜色以处理缺失值
        self.color_to_idx['未知'] = 0
        self.idx_to_color[0] = '未知'
        
    def fit(self, df):
        """
        学习数据的统计特性和构建颜色词汇表

        参数:
            df: DataFrame, 包含材料数据的数据框

        返回:
            self: 当前预处理器实例
        """
        # 1. 处理颜色文本 - 构建颜色词汇表
        # 确保颜色列存在
        if '颜色' not in df.columns:
            print("警告: 数据中没有'颜色'列，将使用默认值")
            return self
            
        # 填充缺失的颜色值
        df['颜色'] = df['颜色'].fillna('未知')
        
        # 构建颜色词汇表
        unique_colors = df['颜色'].unique().tolist()
        # 确保'未知'在词汇表中
        if '未知' not in unique_colors:
            unique_colors.append('未知')
            
        # 构建映射
        for idx, color in enumerate(unique_colors):
            if idx == 0 and color != '未知':
                # 保留索引0给'未知'
                self.color_to_idx[color] = idx + 1
                self.idx_to_color[idx + 1] = color
            else:
                self.color_to_idx[color] = idx
                self.idx_to_color[idx] = color

        # 2. 标准化数值特征 - 排除颜色、标签和独热编码列
        numerical_columns = [
            col for col in df.columns if col not in ['名称', '颜色', 'label'] 
            and col not in [str(i) for i in range(2, 106)]  # 排除独热编码列(2-105)
        ]
        
        print(f"数值特征列: {numerical_columns}")
        
        # 使用-1填充缺失值，并拟合数据
        numerical_data = df[numerical_columns].copy()
        
        # 处理非数值数据，如包含文本字符的列
        for col in numerical_data.columns:
            if numerical_data[col].dtype == 'object':
                print(f"尝试转换列 '{col}' 为数值型")
                # 尝试将字符串转换为数值，失败则用-1填充
                numerical_data[col] = pd.to_numeric(numerical_data[col], errors='coerce')
            
        # 填充缺失值
        numerical_data = numerical_data.fillna(-1)
        
        # 拟合标准化器
        self.numerical_scaler.fit(numerical_data)

        return self

    def transform(self, df):
        """转换数据为模型所需格式"""
        # 1. 处理颜色文本
        # 确保颜色列存在
        if '颜色' not in df.columns:
            print("警告: 变换数据中没有'颜色'列，使用默认值")
            color_indices = np.zeros(len(df), dtype=int)
        else:
            # 填充缺失的颜色值
            df['颜色'] = df['颜色'].fillna('未知')
            # 将未知颜色映射到0
            color_indices = df['颜色'].map(lambda x: self.color_to_idx.get(x, 0)).values

        # 2. 处理数值特征
        numerical_columns = [
            col for col in df.columns if col not in ['名称', '颜色', 'label']
            and col not in [str(i) for i in range(2, 106)]  # 排除独热编码列(2-105)
        ]
        
        # 检查列是否存在
        existing_columns = [col for col in numerical_columns if col in df.columns]
        missing_columns = [col for col in numerical_columns if col not in df.columns]
        
        if missing_columns:
            print(f"警告: 未找到以下数值特征列: {missing_columns}")
            
        # 准备数值特征数据
        numerical_data = pd.DataFrame(index=df.index)
        
        for col in numerical_columns:
            if col in df.columns:
                # 复制列数据
                numerical_data[col] = df[col].copy()
                
                # 处理非数值数据
                if numerical_data[col].dtype == 'object':
                    numerical_data[col] = pd.to_numeric(numerical_data[col], errors='coerce')
            else:
                # 如果列不存在，创建一个全为-1的列
                numerical_data[col] = -1
                
        # 填充缺失值
        df_numerical = numerical_data.fillna(-1)

        # 检测异常值 (Z-score > 3)
        z_scores = np.abs((df_numerical - df_numerical.mean()) / df_numerical.std().replace(0, 1))
        df_numerical = df_numerical.mask(z_scores > 3, -1)  # 将异常值替换为-1

        # 标准化
        numerical_features = self.numerical_scaler.transform(df_numerical)

        # 3. 获取标签（如果存在）
        labels = None
        if 'label' in df.columns:
            labels = df['label'].values

        return color_indices, numerical_features, labels

    def get_vocab_size(self):
        return len(self.color_to_idx)


class MaterialDataset(Dataset):
    def __init__(self, color_indices, numerical_features, labels=None):
        self.color_indices = torch.tensor(color_indices, dtype=torch.long)
        self.numerical_features = torch.tensor(
            numerical_features, dtype=torch.float32)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self):
        return len(self.color_indices)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.color_indices[idx], self.numerical_features[idx], self.labels[idx]
        else:
            return self.color_indices[idx], self.numerical_features[idx]


def create_dataloaders(df, preprocessor, batch_size=32, train_ratio=0.8, val_ratio=0.1, random_state=42):
    """创建训练、验证和测试数据加载器"""
    # 预处理数据
    preprocessor.fit(df)
    color_indices, numerical_features, labels = preprocessor.transform(df)

    # 划分数据集
    n_samples = len(df)

    # 使用指定的随机种子以确保可重复性
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n_samples)

    train_end = int(train_ratio * n_samples)
    val_end = int((train_ratio + val_ratio) * n_samples)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # 创建数据集
    train_dataset = MaterialDataset(
        color_indices[train_indices],
        numerical_features[train_indices],
        labels[train_indices] if labels is not None else None
    )

    val_dataset = MaterialDataset(
        color_indices[val_indices],
        numerical_features[val_indices],
        labels[val_indices] if labels is not None else None
    )

    test_dataset = MaterialDataset(
        color_indices[test_indices],
        numerical_features[test_indices],
        labels[test_indices] if labels is not None else None
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
