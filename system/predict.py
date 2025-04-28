import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 设置matplotlib支持中文显示
def set_chinese_font():
    """设置中文字体支持"""
    # 尝试设置中文字体，按优先级尝试不同字体
    font_list = ['SimHei', 'Microsoft YaHei', 'STXihei', 'STHeiti', 'FangSong', 'KaiTi']
    
    # 检查系统上是否有可用的中文字体
    chinese_font = None
    for font in font_list:
        try:
            font_prop = FontProperties(fname=mpl.font_manager.findfont(font))
            chinese_font = font
            print(f"已找到并使用中文字体: {chinese_font}")
            break
        except:
            continue
    
    if chinese_font:
        plt.rcParams['font.family'] = chinese_font
    else:
        # 如果没有中文字体，使用无衬线字体，可能仍然不能很好地显示中文
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
        print("警告: 未找到合适的中文字体，中文可能无法正常显示")
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

from model import load_model
from data_preprocessing import load_preprocessor

def predict_material(features, model, material_names, class_indices=None):
    """
    预测材料类型
    
    参数:
    - features: 预处理后的特征张量
    - model: 加载的模型
    - material_names: 材料名称列表
    - class_indices: 类别索引映射字典
    
    返回:
    - top_k_preds: 前k个预测结果 (索引, 概率, 材料名称)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    features = features.to(device)
    
    # 进行预测
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
    
    # 获取预测结果
    probabilities = probabilities.cpu().numpy()[0]
    
    # 获取前k个预测结果
    k = min(5, len(probabilities))
    top_k_indices = np.argsort(probabilities)[::-1][:k]
    top_k_probs = probabilities[top_k_indices]
    
    # 使用类别索引映射来获取材料名称
    top_k_preds = []
    if class_indices:
        # 创建索引到名称的反向映射
        idx_to_name = {v: k for k, v in class_indices.items()}
        for idx, prob in zip(top_k_indices, top_k_probs):
            name = idx_to_name.get(idx, f"未知类别 {idx}")
            top_k_preds.append((idx, prob, name))
    else:
        # 如果没有提供类别索引映射，使用材料名称列表（如果索引有效）
        for idx, prob in zip(top_k_indices, top_k_probs):
            if idx < len(material_names):
                name = material_names[idx]
            else:
                name = f"未知类别 {idx}"
            top_k_preds.append((idx, prob, name))
    
    return top_k_preds

def plot_prediction_results(top_k_preds, save_path=None):
    """
    可视化预测结果
    
    参数:
    - top_k_preds: 前k个预测结果 (索引, 概率, 材料名称)
    - save_path: 保存路径，如果为None则显示图像
    """
    # 设置中文字体
    set_chinese_font()
    
    material_names = [pred[2] for pred in top_k_preds]
    probabilities = [pred[1] for pred in top_k_preds]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(material_names, probabilities, color='cornflowerblue')
    plt.xlabel('概率值', fontsize=12)
    plt.ylabel('材料名称', fontsize=12)
    plt.title('材料分类预测结果', fontsize=14)
    plt.xlim(0, 1)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # 在条形上添加数值标签
    for bar, prob in zip(bars, probabilities):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{prob:.4f}', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def get_user_input_with_missing_values():
    """
    获取用户输入，允许部分特征为空
    
    返回:
    - sample: 包含材料特征的字典，缺失值用None表示
    """
    # 定义所有特征
    features = {
        '颜色': '颜色 (如：金黄色)',
        '密度(g/cm3)': '密度 g/cm3',
        '电阻率': '电阻率',
        '比热容': '比热容',
        '熔点': '熔点',
        '沸点': '沸点',
        '屈服强度': '屈服强度',
        '抗拉强度': '抗拉强度',
        '延展率': '延展率',
        '热膨胀系数': '热膨胀系数',
        '热值(J/kg)': '热值 J/kg',
        '杨氏模量GPa': '杨氏模量 GPa',
        '硬度': '硬度',
        '疲劳强度': '疲劳强度',
        '冲击韧性J/cm2': '冲击韧性 J/cm2'
    }
    
    sample = {}
    
    print("\n请输入已知的材料特征（未知的直接按回车跳过）:")
    
    # 获取颜色 - 必须填写的特征
    while True:
        color = input("颜色 (如：金黄色，必填): ")
        if color.strip():
            sample['颜色'] = color
            break
        else:
            print("颜色是必填项，请输入材料的颜色!")
    
    # 获取其他特征 - 可以为空
    for key, prompt in features.items():
        if key == '颜色':  # 跳过已填写的颜色
            continue
            
        value = input(f"{prompt} (可选，按回车跳过): ")
        if value.strip():
            try:
                sample[key] = float(value)
            except ValueError:
                print(f"警告: '{prompt}'需要数值，输入已被忽略")
    
    # 显示用户输入的特征数量
    filled_features = len(sample)
    total_features = len(features)
    print(f"\n已填写 {filled_features}/{total_features} 个特征")
    
    return sample

def main():
    """主函数"""
    # 设置中文显示
    set_chinese_font()
    
    # 加载材料名称列表
    import pickle
    material_names_path = 'models/material_names.pkl'
    class_indices_path = 'models/class_indices.pkl'
    
    if os.path.exists(material_names_path):
        with open(material_names_path, 'rb') as f:
            material_names = pickle.load(f)
    else:
        print("错误：找不到材料名称列表文件，请先运行训练脚本。")
        return
    
    # 加载类别索引映射（如果存在）
    class_indices = None
    if os.path.exists(class_indices_path):
        with open(class_indices_path, 'rb') as f:
            class_indices = pickle.load(f)
        print(f"已加载类别索引映射，共有 {len(class_indices)} 个映射。")
    else:
        print("警告：找不到类别索引映射文件，使用简单的索引顺序。")
    
    # 加载预处理器和模型
    try:
        preprocessor = load_preprocessor()
        
        # 加载模型
        input_dim = 15  # 1个颜色特征 + 14个数值特征
        hidden_dims = [256, 128, 64]
        num_classes = 104  # 根据数据集中可能的类别数量
        model = load_model(input_dim, hidden_dims, num_classes)
        
        print("模型加载成功！")
    except Exception as e:
        print(f"错误：加载模型或预处理器失败：{e}")
        return
    
    # 获取用户输入
    while True:
        choice = input("\n选择输入方式：\n1. 手动输入（支持部分特征）\n2. 使用示例数据\n3. 退出\n请选择(1/2/3): ")
        
        if choice == '1':
            # 用户手动输入
            sample = get_user_input_with_missing_values()
        elif choice == '2':
            # 使用示例数据 (铜的属性)
            sample = {
                '颜色': '橙红色',
                '密度(g/cm3)': 8.96,
                '电阻率': 1.678,
                '比热容': 0.39,
                '熔点': 1083,
                '沸点': 2567,
                '屈服强度': 220,
                '抗拉强度': 240,
                '延展率': 35,
                '热膨胀系数': 1.485e-05,
                '热值(J/kg)': 24160,
                '杨氏模量GPa': 130,
                '硬度': 40,
                '疲劳强度': 139.3,
                '冲击韧性J/cm2': 42.5
            }
            print("\n使用示例数据：铜")
        elif choice == '3':
            print("退出程序。")
            break
        else:
            print("无效选择，请重新输入。")
            continue
        
        # 预处理输入样本
        features = preprocessor.preprocess_single_sample(sample)
        
        # 进行预测
        top_k_preds = predict_material(features, model, material_names, class_indices)
        
        # 显示结果
        print("\n预测结果:")
        for i, (idx, prob, name) in enumerate(top_k_preds):
            print(f"Top {i+1}: {name} (置信度: {prob:.4f}, 类别索引: {idx})")
        
        # 可视化预测结果
        plot_prediction_results(top_k_preds)

if __name__ == "__main__":
    main()
