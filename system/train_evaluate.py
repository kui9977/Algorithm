import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os
import time
from tqdm import tqdm
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

# 设置matplotlib支持中文显示
def set_chinese_font():
    """设置中文字体支持"""
    # 尝试设置中文字体，按优先级尝试不同字体
    font_list = ['SimHei', 'Microsoft YaHei', 'STXihei', 'STHeiti', 'FangSong', 'KaiTi', 'Arial Unicode MS', 'NSimSun']
    
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
        # 如果没有中文字体，使用无衬线字体，并启用Unicode支持
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
        print("警告: 未找到合适的中文字体，尝试使用默认Unicode兼容字体")
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

from model import MetalClassifier, save_model
from data_preprocessing import DataPreprocessor, save_preprocessor

def train_model(train_loader, val_loader, input_dim, num_classes, hidden_dims=[256, 128, 64], 
                lr=0.001, epochs=100, patience=10, model_path='models/metal_classifier.pth'):
    """
    训练金属分类模型
    
    参数:
    - train_loader: 训练数据加载器
    - val_loader: 验证数据加载器
    - input_dim: 输入特征维度
    - num_classes: 类别数量
    - hidden_dims: 隐藏层维度列表
    - lr: 学习率
    - epochs: 训练轮数
    - patience: 早停耐心值
    - model_path: 模型保存路径
    
    返回:
    - model: 训练好的模型
    - history: 训练历史记录
    """
    # 创建模型
    model = MetalClassifier(input_dim, hidden_dims, num_classes)
    
    # 使用GPU如果可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 用于记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # 早停机制
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 训练循环
    print("开始训练...")
    start_time = time.time()
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录损失和准确率
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 记录损失和准确率
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 检查早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    total_time = time.time() - start_time
    print(f"训练完成! 总用时: {total_time:.2f} 秒")
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 保存模型
    save_model(model, model_path)
    
    return model, history

def evaluate_model(model, val_loader, material_names, class_indices=None):
    """
    评估模型性能
    
    参数:
    - model: 训练好的模型
    - val_loader: 验证数据加载器
    - material_names: 材料名称列表
    - class_indices: 用于映射类别索引到材料名称的字典
    
    返回:
    - eval_metrics: 评估指标
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # 用于存储所有预测结果和真实标签
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    # 获取唯一的类别标签
    unique_classes = sorted(list(set(all_targets)))
    num_classes = len(unique_classes)
    
    # 如果提供了类别索引映射，则使用它来获取材料名称
    label_names = []
    if class_indices is not None:
        # 反转字典，使其从索引映射到材料名称
        idx_to_name = {v: k for k, v in class_indices.items()}
        for idx in unique_classes:
            if idx in idx_to_name:
                label_names.append(idx_to_name[idx])
            else:
                label_names.append(f"Class {idx}")
    else:
        # 如果没有提供映射，使用前N个材料名称（如果有足够的）
        if len(material_names) >= num_classes:
            label_names = [material_names[i] for i in unique_classes]
        else:
            label_names = [f"Class {i}" for i in unique_classes]
    
    # 计算分类报告，使用适当的标签
    report = classification_report(all_targets, all_preds, target_names=label_names, output_dict=True)
    
    # 将分类报告转换为DataFrame
    report_df = pd.DataFrame(report).transpose()
    
    print(f"评估类别数量: {num_classes}")
    print(f"使用的标签数量: {len(label_names)}")
    
    return {
        'confusion_matrix': cm,
        'classification_report': report_df,
        'accuracy': report['accuracy'],
        'unique_classes': unique_classes,
        'label_names': label_names
    }

def plot_learning_curves(history, save_path='learning_curves.png'):
    """
    绘制学习曲线
    
    参数:
    - history: 训练历史记录
    - save_path: 图像保存路径
    """
    # 设置中文字体
    set_chinese_font()
    
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(cm, label_names, save_path='confusion_matrix.png'):
    """
    绘制混淆矩阵
    
    参数:
    - cm: 混淆矩阵
    - label_names: 标签名称列表
    - save_path: 图像保存路径
    """
    # 设置中文字体
    set_chinese_font()
    
    # 如果类别太多，只显示少量类别
    max_display = 20
    if len(label_names) > max_display:
        print(f"混淆矩阵类别过多({len(label_names)}个)，只显示前{max_display}个类别")
        cm = cm[:max_display, :max_display]
        label_names = label_names[:max_display]
    
    # 确保标签名称是字符串类型
    label_names = [str(name).strip() for name in label_names]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主函数"""
    # 数据路径
    train_path = 'd:\\Projects\\Python_projects\\dpl\\Algorithm\\system\\complete_materials_train.csv'
    val_path = 'd:\\Projects\\Python_projects\\dpl\\Algorithm\\system\\complete_materials_val.csv'
    
    # 创建数据预处理器
    print("加载并预处理数据...")
    preprocessor = DataPreprocessor()
    train_loader, val_loader, input_dim, num_classes, material_names, class_indices = preprocessor.load_and_preprocess_data(
        train_path, val_path, batch_size=32
    )
    
    # 保存预处理器
    save_preprocessor(preprocessor)
    
    # 定义隐藏层维度
    hidden_dims = [256, 128, 64]
    
    # 训练模型
    model, history = train_model(
        train_loader, val_loader, input_dim, num_classes, 
        hidden_dims=hidden_dims, lr=0.001, epochs=100, patience=15
    )
    
    # 评估模型
    print("评估模型...")
    eval_metrics = evaluate_model(model, val_loader, material_names, class_indices)
    
    # 打印评估结果
    print(f"模型准确率: {eval_metrics['accuracy']:.4f}")
    print("\n分类报告:")
    print(eval_metrics['classification_report'])
    
    # 绘制学习曲线和混淆矩阵
    print("生成可视化图表...")
    plot_learning_curves(history)
    plot_confusion_matrix(eval_metrics['confusion_matrix'], eval_metrics['label_names'])
    
    print("训练和评估完成!")
    
    return model, preprocessor, material_names, class_indices

if __name__ == "__main__":
    main()
