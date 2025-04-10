import torch
import pandas as pd
import numpy as np
from model import MetalClassifier
from data_preprocessing import preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns

def predict_material(model, features, color_encoder, materials_list, device='cuda'):
    """
    预测材料种类
    
    参数:
        model: 训练好的模型
        features: 材料特征
        color_encoder: 颜色编码器
        materials_list: 材料名称列表
        device: 使用设备
        
    返回:
        预测结果，包括预测的材料名称和概率
    """
    model.eval()
    
    # 将输入转换为PyTorch张量
    input_tensor = torch.tensor(features.values, dtype=torch.float32).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        # 获取最可能的材料类别
        _, predicted = torch.max(outputs, 1)
        
    # 获取预测的类别和概率
    predictions = predicted.cpu().numpy()
    prob_values = probabilities.cpu().numpy()
    
    results = []
    for i in range(len(predictions)):
        material_idx = predictions[i]
        material_name = materials_list[material_idx]
        probability = prob_values[i, material_idx]
        
        # 获取前5个最可能的材料类别
        top5_indices = np.argsort(prob_values[i])[::-1][:5]
        top5_materials = [(materials_list[idx], prob_values[i, idx]) for idx in top5_indices]
        
        results.append({
            'predicted_material': material_name,
            'probability': probability,
            'top5_predictions': top5_materials
        })
    
    return results

def plot_prediction_probabilities(results, index=0):
    """
    绘制预测概率分布图
    
    参数:
        results: 预测结果
        index: 要显示的结果索引
    """
    # 获取指定索引的结果
    result = results[index]
    top5_predictions = result['top5_predictions']
    
    # 提取材料名称和概率
    materials = [item[0] for item in top5_predictions]
    probabilities = [item[1] for item in top5_predictions]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(materials, probabilities, color='skyblue')
    
    # 添加数值标签
    for bar, prob in zip(bars, probabilities):
        plt.text(bar.get_x() + bar.get_width()/2, 
                 prob + 0.01, 
                 f'{prob:.2f}', 
                 ha='center', 
                 va='bottom')
    
    plt.xlabel('Materials')
    plt.ylabel('Probability')
    plt.title('Top 5 Predicted Materials')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()

def load_trained_model(model_path, input_size, num_classes, device='cuda'):
    """
    加载训练好的模型
    
    参数:
        model_path: 模型路径
        input_size: 输入特征维度
        num_classes: 类别数量
        device: 使用设备
        
    返回:
        加载好的模型
    """
    model = MetalClassifier(input_size, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载训练和验证数据以获取预处理器和材料列表
    train_path = "d:/Projects/Python_projects/dpl/Algorithm/system/complete_materials_train.csv"
    val_path = "d:/Projects/Python_projects/dpl/Algorithm/system/complete_materials_val.csv"
    
    # 加载训练数据和验证数据
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    
    # 获取材料名称列表
    materials_list = train_data.iloc[:, 0].tolist()
    
    # 示例：预测验证集中的样本
    val_sample = val_data.iloc[0:1, :]  # 取第一个样本进行预测
    
    # 处理特征
    sample_features = val_sample.iloc[:, 105:].copy()
    sample_features['name'] = val_sample.iloc[:, 0]
    
    # 加载训练好的模型
    model_path = "d:/Projects/Python_projects/dpl/Algorithm/system/models/metal_classifier.pt"
    input_size = 15 + len(set(train_data['颜色'].dropna()))  # 数值特征 + 颜色独热编码维度
    num_classes = len(materials_list)
    
    model = load_trained_model(model_path, input_size, num_classes, device)
    
    # 预处理样本特征
    _, sample_features_processed, color_encoder = preprocess_data(
        pd.concat([train_data.iloc[:, 105:], pd.DataFrame(columns=['name'])], axis=1),
        sample_features
    )
    
    # 删除名称列
    sample_features_processed = sample_features_processed.drop('name', axis=1)
    
    # 进行预测
    prediction_results = predict_material(model, sample_features_processed, color_encoder, materials_list, device)
    
    # 打印预测结果
    for i, result in enumerate(prediction_results):
        print(f"样本 {i+1}:")
        print(f"预测材料: {result['predicted_material']}")
        print(f"预测概率: {result['probability']:.4f}")
        print("Top 5 预测:")
        for material, prob in result['top5_predictions']:
            print(f"  {material}: {prob:.4f}")
        print()
    
    # 绘制预测概率分布图
    plot_prediction_probabilities(prediction_results)
