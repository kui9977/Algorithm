import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

from data_preprocessing import load_data, preprocess_data
from model import MetalClassifier, EnsembleModel

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, device='cuda', patience=15):
    """
    训练模型
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 训练设备 ('cuda' 或 'cpu')
        patience: 早停的耐心值，验证损失不再减小的轮数
        
    返回:
        训练好的模型和训练历史
    """
    # 将模型移至指定设备
    model = model.to(device)
    
    # 用于记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # 用于早停
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 使用tqdm显示进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            # 更新进度条
            train_pbar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total})
        
        # 计算训练集上的平均损失和准确率
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # 禁用梯度计算，节省内存
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                
                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # 计算验证集上的平均损失和准确率
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # 记录训练历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 打印每轮的结果
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # 早停策略
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
            
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # 如果有最佳模型状态，则加载它
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, history

def evaluate(model, data_loader, criterion, device='cuda', material_names=None):
    """
    评估模型
    
    参数:
        model: 训练好的模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 评估设备
        material_names: 材料名称列表，用于显示分类报告
        
    返回:
        损失，准确率，预测标签，真实标签，预测概率
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader.dataset)
    
    # 计算准确率
    accuracy = accuracy_score(all_targets, all_predictions)
    
    # 打印详细的分类报告
    if material_names is not None:
        # 获取类别标签
        target_names = [material_names[i] for i in range(len(material_names))]
        print(classification_report(all_targets, all_predictions, target_names=target_names))
    
    return avg_loss, accuracy, all_predictions, all_targets, all_probabilities

def plot_learning_curves(history, save_path=None):
    """
    绘制学习曲线
    
    参数:
        history: 训练历史数据
        save_path: 保存图像的路径
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_confusion_matrix(true_labels, pred_labels, class_names, save_path=None):
    """
    绘制混淆矩阵
    
    参数:
        true_labels: 真实标签
        pred_labels: 预测标签
        class_names: 类别名称
        save_path: 保存图像的路径
    """
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 如果类别太多，可能需要显示部分类别
    if len(class_names) > 20:
        # 找出最频繁的20个类别
        class_counts = np.sum(cm, axis=1)
        top_indices = np.argsort(class_counts)[-20:]
        cm = cm[top_indices, :][:, top_indices]
        selected_classes = [class_names[i] for i in top_indices]
    else:
        selected_classes = class_names
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=selected_classes, 
                yticklabels=selected_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def save_model(model, path):
    """
    保存模型
    
    参数:
        model: 要保存的模型
        path: 保存路径
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), path)
    print(f"模型已保存到 {path}")

def load_model(model, path, device='cuda'):
    """
    加载模型
    
    参数:
        model: 模型结构
        path: 加载路径
        device: 设备
        
    返回:
        加载的模型
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

if __name__ == "__main__":
    # 设置随机种子以确保可复现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载和预处理数据
    train_path = "d:/Projects/Python_projects/dpl/Algorithm/system/complete_materials_train.csv"
    val_path = "d:/Projects/Python_projects/dpl/Algorithm/system/complete_materials_val.csv"
    
    train_features, train_labels, val_features, val_labels = load_data(train_path, val_path)
    train_features, val_features, color_encoder = preprocess_data(train_features, val_features)
    
    # 保存材料名称用于后续评估
    material_names = train_features['name'].tolist()
    
    # 删除名称列
    train_features = train_features.drop('name', axis=1)
    val_features = val_features.drop('name', axis=1)
    
    # 转换为PyTorch张量
    train_features_tensor = torch.tensor(train_features.values, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    
    val_features_tensor = torch.tensor(val_features.values, dtype=torch.float32)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
    
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_features_tensor, val_labels_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 初始化模型
    input_size = train_features.shape[1]
    num_classes = len(np.unique(np.concatenate([train_labels, val_labels])))
    
    print(f"输入特征维度: {input_size}")
    print(f"类别数量: {num_classes}")
    
    # 创建单个模型
    model = MetalClassifier(input_size, num_classes, hidden_size=256)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 训练模型
    trained_model, history = train(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer,
        num_epochs=100,
        device=device,
        patience=15
    )
    
    # 评估模型
    val_loss, val_accuracy, val_predictions, val_targets, val_probabilities = evaluate(
        trained_model, 
        val_loader, 
        criterion, 
        device=device,
        material_names=material_names
    )
    
    print(f"验证集损失: {val_loss:.4f}")
    print(f"验证集准确率: {val_accuracy:.4f}")
    
    # 绘制学习曲线
    plot_learning_curves(history, save_path="d:/Projects/Python_projects/dpl/Algorithm/system/learning_curves.png")
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        val_targets, 
        val_predictions, 
        material_names, 
        save_path="d:/Projects/Python_projects/dpl/Algorithm/system/confusion_matrix.png"
    )
    
    # 保存模型
    save_model(trained_model, "d:/Projects/Python_projects/dpl/Algorithm/system/models/metal_classifier.pt")
