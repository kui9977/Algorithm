import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(train_path, val_path):
    """
    加载训练集和验证集数据
    
    参数:
        train_path: 训练集文件路径
        val_path: 验证集文件路径
    
    返回:
        训练特征, 训练标签, 验证特征, 验证标签
    """
    # 加载训练数据和验证数据
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    
    # 分离特征和标签
    # 第2-105列是独热编码的标签
    train_labels = train_data.iloc[:, 1:105].values
    val_labels = val_data.iloc[:, 1:105].values
    
    # 将独热编码转换为类别索引
    train_labels = np.argmax(train_labels, axis=1)
    val_labels = np.argmax(val_labels, axis=1)
    
    # 获取特征列，从第106列开始
    train_features = train_data.iloc[:, 105:].copy()
    val_features = val_data.iloc[:, 105:].copy()
    
    # 添加名称列，用于后续参考
    train_features['name'] = train_data.iloc[:, 0]
    val_features['name'] = val_data.iloc[:, 0]
    
    return train_features, train_labels, val_features, val_labels

def preprocess_data(train_features, val_features):
    """
    预处理特征数据，包括处理分类特征、标准化数值特征等
    
    参数:
        train_features: 训练集特征
        val_features: 验证集特征
    
    返回:
        预处理后的训练特征和验证特征
    """
    # 保存材料名称，稍后需要
    train_names = train_features['name']
    val_names = val_features['name']
    
    # 删除名称列，因为它不是用于训练的特征
    train_features = train_features.drop('name', axis=1)
    val_features = val_features.drop('name', axis=1)
    
    # 提取颜色，这是唯一的分类特征
    color_encoder = LabelEncoder()
    
    # 合并训练和验证集的颜色进行编码，确保覆盖所有可能的颜色
    all_colors = pd.concat([train_features['颜色'], val_features['颜色']])
    color_encoder.fit(all_colors.dropna())
    
    # 编码颜色特征
    train_features['颜色'] = color_encoder.transform(train_features['颜色'].fillna('未知'))
    val_features['颜色'] = color_encoder.transform(val_features['颜色'].fillna('未知'))
    
    # 用平均值填充数值特征中的缺失值
    for column in train_features.columns:
        if column != '颜色':
            mean_value = train_features[column].mean()
            train_features[column].fillna(mean_value, inplace=True)
            val_features[column].fillna(mean_value, inplace=True)
    
    # 标准化数值特征
    scaler = StandardScaler()
    
    # 获取除颜色外的所有数值特征
    numerical_cols = [col for col in train_features.columns if col != '颜色']
    
    # 对数值特征进行标准化
    train_features[numerical_cols] = scaler.fit_transform(train_features[numerical_cols])
    val_features[numerical_cols] = scaler.transform(val_features[numerical_cols])
    
    # 创建颜色的独热编码
    color_one_hot = pd.get_dummies(train_features['颜色'], prefix='color')
    val_color_one_hot = pd.get_dummies(val_features['颜色'], prefix='color')
    
    # 确保验证集具有与训练集相同的颜色编码列
    for col in color_one_hot.columns:
        if col not in val_color_one_hot.columns:
            val_color_one_hot[col] = 0
    
    # 确保列的顺序一致
    val_color_one_hot = val_color_one_hot[color_one_hot.columns]
    
    # 删除原始颜色列并添加独热编码列
    train_features = train_features.drop('颜色', axis=1)
    val_features = val_features.drop('颜色', axis=1)
    
    train_features = pd.concat([train_features, color_one_hot], axis=1)
    val_features = pd.concat([val_features, val_color_one_hot], axis=1)
    
    # 添加材料名称回来，方便调试
    train_features['name'] = train_names
    val_features['name'] = val_names
    
    return train_features, val_features, color_encoder

if __name__ == "__main__":
    # 测试数据加载和预处理
    train_path = "d:/Projects/Python_projects/dpl/Algorithm/system/complete_materials_train.csv"
    val_path = "d:/Projects/Python_projects/dpl/Algorithm/system/complete_materials_val.csv"
    
    train_features, train_labels, val_features, val_labels = load_data(train_path, val_path)
    train_features, val_features, color_encoder = preprocess_data(train_features, val_features)
    
    print(f"训练集形状: {train_features.shape}")
    print(f"验证集形状: {val_features.shape}")
    print(f"颜色种类: {len(color_encoder.classes_)}")
