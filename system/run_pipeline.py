import os
import torch
import time
from tqdm import tqdm
import sys

# 设置系统默认编码
if sys.version_info >= (3, 0):
    import importlib
    importlib.reload(sys)
else:
    reload(sys)
    sys.setdefaultencoding('utf-8')

# 导入自定义模块
from data_preprocessing import DataPreprocessor, save_preprocessor
from model import MetalClassifier, save_model
from train_evaluate import train_model, evaluate_model, plot_learning_curves, plot_confusion_matrix, set_chinese_font
from save_material_names import extract_and_save_material_names

def check_file_encoding(filepath):
    """检查文件编码"""
    try:
        import chardet
        with open(filepath, 'rb') as f:
            raw_data = f.read(10000)  # 读取前10000字节
            result = chardet.detect(raw_data)
            print(f"文件 {filepath} 检测到的编码: {result['encoding']} (置信度: {result['confidence']})")
        return result
    except ImportError:
        print("未安装chardet库，无法检测文件编码")
        return None
    except Exception as e:
        print(f"检测文件编码时出错: {e}")
        return None

def run_pipeline():
    """运行完整的金属材料分类算法流程"""
    print("=== 金属材料多模态识别系统 ===")
    
    # 设置随机种子以获得可复现的结果
    torch.manual_seed(42)
    
    # 数据路径
    train_path = 'd:\\Projects\\Python_projects\\dpl\\Algorithm\\system\\complete_materials_train.csv'
    val_path = 'd:\\Projects\\Python_projects\\dpl\\Algorithm\\system\\complete_materials_val.csv'
    
    # 检查文件编码
    print("\n检查训练和验证数据文件编码...")
    check_file_encoding(train_path)
    check_file_encoding(val_path)
    
    try:
        # 1. 提取并保存材料名称
        print("\n步骤 1: 提取并保存材料名称...")
        material_names = extract_and_save_material_names()
        
        # 2. 数据预处理
        print("\n步骤 2: 加载并预处理数据...")
        preprocessor = DataPreprocessor()
        train_loader, val_loader, input_dim, num_classes, material_names, class_indices = preprocessor.load_and_preprocess_data(
            train_path, val_path, batch_size=32
        )
        
        # 保存预处理器
        save_preprocessor(preprocessor)
        print(f"预处理器已保存到 models/preprocessor.pkl")
        
        # 保存类别索引映射
        import pickle
        with open('models/class_indices.pkl', 'wb') as f:
            pickle.dump(class_indices, f)
        print(f"类别索引映射已保存到 models/class_indices.pkl")
        
        # 3. 模型训练
        print("\n步骤 3: 训练模型...")
        # 定义隐藏层维度
        hidden_dims = [256, 128, 64]
        
        # 训练模型
        model, history = train_model(
            train_loader, val_loader, input_dim, num_classes, 
            hidden_dims=hidden_dims, lr=0.001, epochs=100, patience=15
        )
        
        # 4. 模型评估
        print("\n步骤 4: 评估模型性能...")
        eval_metrics = evaluate_model(model, val_loader, material_names, class_indices)
        
        # 打印评估结果
        print(f"\n模型准确率: {eval_metrics['accuracy']:.4f}")
        print("\n分类报告:")
        print(eval_metrics['classification_report'])
        
        # 5. 可视化结果
        print("\n步骤 5: 生成可视化图表...")
        plot_learning_curves(history)
        plot_confusion_matrix(eval_metrics['confusion_matrix'], eval_metrics['label_names'])
        
        # 6. 示例预测（使用部分特征）
        print("\n步骤 6: 预测演示 - 使用部分特征...")
        from predict import predict_material, plot_prediction_results
        
        # 获取训练集中已知的颜色列表
        known_colors = preprocessor.get_known_colors()
        print(f"训练集中包含的颜色: {known_colors}")
        
        # 选择一个训练集中已知的颜色
        known_color = "紫红色"  # 这是铜的颜色，在训练集中存在
        
        # 示例：只有颜色和密度的样本
        partial_sample = {
            '颜色': known_color,
            '密度(g/cm3)': 8.96
        }
        
        print("\n使用部分特征进行预测:")
        print(f"已知特征: 颜色={partial_sample['颜色']}, 密度={partial_sample['密度(g/cm3)']} g/cm3")
        
        features = preprocessor.preprocess_single_sample(partial_sample)
        top_k_preds = predict_material(features, model, material_names, class_indices)
        
        print("\n预测结果:")
        for i, (idx, prob, name) in enumerate(top_k_preds[:3]):  # 只显示前3个结果
            print(f"Top {i+1}: {name} (置信度: {prob:.4f})")
        
        # 保存预处理器信息
        if hasattr(preprocessor, 'successful_encoding') and preprocessor.successful_encoding:
            encoding_info_path = 'models/encoding_info.txt'
            with open(encoding_info_path, 'w', encoding='utf-8') as f:
                f.write(f"数据文件成功使用的编码: {preprocessor.successful_encoding}\n")
                f.write(f"可用颜色列表: {', '.join(known_colors)}\n")
            print(f"编码信息已保存到 {encoding_info_path}")
        
        # 尝试使用一个未知颜色进行预测（测试容错处理）
        unknown_color_sample = {
            '颜色': '橙红色',  # 未知颜色
            '密度(g/cm3)': 8.96
        }
        
        print("\n尝试使用未知颜色预测:")
        print(f"已知特征: 颜色={unknown_color_sample['颜色']}, 密度={unknown_color_sample['密度(g/cm3)']} g/cm3")
        
        try:
            features = preprocessor.preprocess_single_sample(unknown_color_sample)
            top_k_preds = predict_material(features, model, material_names, class_indices)
            
            print("\n预测结果 (使用未知颜色):")
            for i, (idx, prob, name) in enumerate(top_k_preds[:3]):
                print(f"Top {i+1}: {name} (置信度: {prob:.4f})")
        except Exception as e:
            print(f"预测失败: {e}")
        
        print("\n=== 流程完成 ===")
        print(f"模型已保存到 models/metal_classifier.pth")
        print("学习曲线已保存到 learning_curves.png")
        print("混淆矩阵已保存到 confusion_matrix.png")
        
        print("\n现在您可以运行 predict.py 来预测新的材料样本（支持部分特征输入）。")
    
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 提供编码相关建议
        print("\n可能是编码问题，尝试以下解决方案:")
        print("1. 安装chardet库: pip install chardet")
        print("2. 尝试手动将CSV文件转换为UTF-8编码")
        print("3. 使用文本编辑器打开CSV文件，另存为UTF-8编码格式")

if __name__ == "__main__":
    run_pipeline()
