import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

class MetalDataset(Dataset):
    """金属材料数据集"""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DataPreprocessor:
    """数据预处理类"""
    def __init__(self):
        self.color_encoder = LabelEncoder()
        self.num_scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.color_column_idx = 105  # 颜色特征在处理后的索引
        self.successful_encoding = None  # 用于存储成功的编码格式
        
    def load_and_preprocess_data(self, train_path, val_path, batch_size=32):
        """加载并预处理训练和验证数据"""
        # 尝试多种编码方式读取数据
        encodings = ['gbk', 'gb2312', 'gb18030', 'cp936', 'latin-1', 'utf-8-sig', 'iso-8859-1']
        
        train_df = None
        val_df = None
        
        # 尝试多种编码方式读取数据
        for encoding in encodings:
            try:
                print(f"尝试使用 {encoding} 编码读取数据...")
                train_df = pd.read_csv(train_path, encoding=encoding)
                val_df = pd.read_csv(val_path, encoding=encoding)
                self.successful_encoding = encoding
                print(f"成功使用 {encoding} 编码读取数据!")
                break
            except Exception as e:
                print(f"使用 {encoding} 编码读取失败: {e}")
                continue
        
        if train_df is None or val_df is None:
            raise ValueError("无法读取数据文件，请检查文件格式和编码。")
        
        # 提取标签（独热编码转为类别索引）
        train_labels = train_df.iloc[:, 1:105].values.argmax(axis=1)
        val_labels = val_df.iloc[:, 1:105].values.argmax(axis=1)
        
        # 提取特征
        train_features, val_features = self._preprocess_features(train_df, val_df)
        
        # 创建DataLoader
        train_dataset = MetalDataset(train_features, train_labels)
        val_dataset = MetalDataset(val_features, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 获取类别数量（材料种类）
        num_classes = train_df.iloc[:, 1:105].shape[1]
        
        # 获取特征维度
        input_dim = train_features.shape[1]
        
        # 构建类别索引到材料名称的映射
        material_names = []
        class_indices = {}
        
        # 遍历数据，创建映射
        for i in range(train_df.shape[0]):
            row = train_df.iloc[i]
            name = row.iloc[0]
            idx = row.iloc[1:105].values.argmax()
            material_names.append(name)
            class_indices[name] = idx
        
        # 然后再检查验证集
        for i in range(val_df.shape[0]):
            row = val_df.iloc[i]
            name = row.iloc[0]
            idx = row.iloc[1:105].values.argmax()
            material_names.append(name)
            class_indices[name] = idx
        
        # 删除重复的材料名称并排序
        material_names = sorted(list(set(material_names)))
        
        print(f"训练+验证集中的类别数量: {len(set(np.concatenate([train_labels, val_labels])))}")
        print(f"材料名称数量: {len(material_names)}")
        print(f"类别映射数量: {len(class_indices)}")
        
        return train_loader, val_loader, input_dim, num_classes, material_names, class_indices
    
    def _preprocess_features(self, train_df, val_df):
        """预处理特征"""
        # 获取颜色特征（分类特征）
        train_color = train_df.iloc[:, 105].values.reshape(-1, 1)  # 颜色列索引为105
        val_color = val_df.iloc[:, 105].values.reshape(-1, 1)
        
        # 编码颜色特征
        self.color_encoder.fit(train_color.ravel())
        train_color_encoded = self.color_encoder.transform(train_color.ravel()).reshape(-1, 1)
        val_color_encoded = self.color_encoder.transform(val_color.ravel()).reshape(-1, 1)
        
        # 获取数值特征（从第106列开始，即索引从106开始）
        train_numeric = train_df.iloc[:, 106:].values
        val_numeric = val_df.iloc[:, 106:].values
        
        # 填充可能的缺失值
        self.imputer.fit(train_numeric)
        train_numeric_imputed = self.imputer.transform(train_numeric)
        val_numeric_imputed = self.imputer.transform(val_numeric)
        
        # 标准化数值特征
        self.num_scaler.fit(train_numeric_imputed)
        train_numeric_scaled = self.num_scaler.transform(train_numeric_imputed)
        val_numeric_scaled = self.num_scaler.transform(val_numeric_imputed)
        
        # 将颜色特征和数值特征合并
        train_features = np.hstack((train_color_encoded, train_numeric_scaled))
        val_features = np.hstack((val_color_encoded, val_numeric_scaled))
        
        return train_features, val_features
    
    def preprocess_single_sample(self, sample_dict):
        """
        预处理单个样本用于预测，支持部分特征缺失
        
        参数:
        - sample_dict: 包含材料特征的字典，缺失值可以不存在
        
        返回:
        - features: 预处理后的特征张量
        """
        # 必须提供颜色
        if '颜色' not in sample_dict:
            raise ValueError("颜色是必须的特征，缺失将无法进行预测")
        
        # 创建颜色特征数组
        color = np.array([sample_dict['颜色']]).reshape(-1, 1)
        
        # 编码颜色
        color_encoded = self.color_encoder.transform(color.ravel()).reshape(-1, 1)
        
        # 准备数值特征默认值字典
        default_values = {
            '密度(g/cm3)': np.nan,
            '电阻率': np.nan,
            '比热容': np.nan,
            '熔点': np.nan,
            '沸点': np.nan,
            '屈服强度': np.nan,
            '抗拉强度': np.nan,
            '延展率': np.nan,
            '热膨胀系数': np.nan,
            '热值(J/kg)': np.nan,
            '杨氏模量GPa': np.nan,
            '硬度': np.nan,
            '疲劳强度': np.nan,
            '冲击韧性J/cm2': np.nan
        }
        
        # 用提供的值更新默认值
        for key in default_values:
            if key in sample_dict:
                default_values[key] = sample_dict[key]
        
        # 创建数值特征数组，允许包含NaN值
        numeric_features = np.array([
            default_values['密度(g/cm3)'],
            default_values['电阻率'],
            default_values['比热容'],
            default_values['熔点'],
            default_values['沸点'],
            default_values['屈服强度'],
            default_values['抗拉强度'],
            default_values['延展率'],
            default_values['热膨胀系数'],
            default_values['热值(J/kg)'],
            default_values['杨氏模量GPa'],
            default_values['硬度'],
            default_values['疲劳强度'],
            default_values['冲击韧性J/cm2']
        ]).reshape(1, -1)
        
        # 检查并显示缺失值比例
        missing_count = np.isnan(numeric_features).sum()
        if missing_count > 0:
            print(f"注意: 有 {missing_count} 个特征缺失，将使用训练集均值填充")
        
        # 填充和标准化数值特征
        numeric_imputed = self.imputer.transform(numeric_features)
        numeric_scaled = self.num_scaler.transform(numeric_imputed)
        
        # 合并所有特征
        features = np.hstack((color_encoded, numeric_scaled))
        
        return torch.tensor(features, dtype=torch.float32)

def save_preprocessor(preprocessor, path='models/preprocessor.pkl'):
    """保存预处理器"""
    import pickle
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # 保存成功的编码格式供将来使用
    if preprocessor.successful_encoding:
        encoding_path = os.path.join(os.path.dirname(path), 'successful_encoding.txt')
        with open(encoding_path, 'w') as f:
            f.write(preprocessor.successful_encoding)
        print(f"成功的编码格式 '{preprocessor.successful_encoding}' 已保存到 {encoding_path}")

def load_preprocessor(path='models/preprocessor.pkl'):
    """加载预处理器"""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)
