import os
import torch
import time
from tqdm import tqdm

# 导入自定义模块
from data_preprocessing import DataPreprocessor, save_preprocessor
from model import MetalClassifier, save_model
from train_evaluate import train_model, evaluate_model, plot_learning_curves, plot_confusion_matrix
from save_material_names import extract_and_save_material_names

def run_pipeline():
    """运行完整的金属材料分类算法流程"""
    print("=== 金属材料多模态识别系统 ===")
    
    # 设置随机种子以获得可复现的结果
    torch.manual_seed(42)
    
    # 数据路径
    train_path = 'd:\\Projects\\Python_projects\\dpl\\Algorithm\\system\\complete_materials_train.csv'
    val_path = 'd:\\Projects\\Python_projects\\dpl\\Algorithm\\system\\complete_materials_val.csv'
    
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
        
        # 示例：只有颜色和密度的样本
        partial_sample = {
            '颜色': '橙红色',
            '密度(g/cm3)': 8.96
        }
        
        print("\n使用部分特征进行预测:")
        print(f"已知特征: 颜色={partial_sample['颜色']}, 密度={partial_sample['密度(g/cm3)']} g/cm3")
        
        features = preprocessor.preprocess_single_sample(partial_sample)
        top_k_preds = predict_material(model, features, material_names, class_indices)
        
        print("\n预测结果:")
        for i, (idx, prob, name) in enumerate(top_k_preds[:3]):  # 只显示前3个结果
            print(f"Top {i+1}: {name} (置信度: {prob:.4f})")
        
        print("\n=== 流程完成 ===")
        print(f"模型已保存到 models/metal_classifier.pth")
        print("学习曲线已保存到 learning_curves.png")
        print("混淆矩阵已保存到 confusion_matrix.png")
        
        print("\n现在您可以运行 predict.py 来预测新的材料样本（支持部分特征输入）。")
    
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline()
