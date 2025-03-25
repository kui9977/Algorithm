import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm

from models.model import MaterialAnalysisModel
from data.preprocessor import MaterialDataPreprocessor, create_dataloaders


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        初始化早停机制

        参数:
            patience: 在多少轮验证损失未改善后停止训练
            min_delta: 被视为改进的最小变化量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        检查是否应该早停

        参数:
            val_loss: 当前验证损失

        当counter达到patience时，early_stop将设为True
        """
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.min_delta:
            self.counter += 1  # 损失增加，计数器加1
            if self.counter >= self.patience:
                self.early_stop = True  # 达到耐心值，触发早停
        else:
            self.best_score = val_loss  # 损失改善，更新最佳分数
            self.counter = 0  # 重置计数器


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=30, device='cuda', patience=5):
    """
    训练模型并在验证集上评估

    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        num_epochs: 训练的最大轮数
        device: 使用的计算设备(GPU/CPU)
        patience: 早停的耐心值

    返回:
        训练好的模型
    """
    # 将模型移到指定设备
    model = model.to(device)

    # 初始化参数
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # 初始化早停
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for color_idx, numerical, labels in pbar:
            # 将数据移到指定设备
            color_idx = color_idx.to(device)
            numerical = numerical.to(device)
            labels = labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(color_idx, numerical)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)

            # 更新进度条
            pbar.set_postfix(
                {'loss': loss.item(), 'acc': train_correct / train_total})

        # 计算平均训练损失和准确率
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for color_idx, numerical, labels in pbar:
                # 将数据移到指定设备
                color_idx = color_idx.to(device)
                numerical = numerical.to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = model(color_idx, numerical)
                loss = criterion(outputs, labels)

                # 统计损失和准确率
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

                # 更新进度条
                pbar.set_postfix(
                    {'loss': loss.item(), 'acc': val_correct / val_total})

        # 计算平均验证损失和准确率
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 更新学习率
        scheduler.step()

        # 打印结果
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print('Saved best model!')

        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print('Early stopping!')
            break

    # 绘制训练过程
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('training_history.png')

    return model


def evaluate_model(model, test_loader, criterion, device='cuda', class_names=None):
    """
    在测试集上评估模型

    参数:
        model: 要评估的模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 使用的计算设备(GPU/CPU)
        class_names: 类别名称列表（可选）

    返回:
        准确率, F1分数, 混淆矩阵
    """
    # 将模型移到指定设备
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    test_loss = 0.0

    with torch.no_grad():
        for color_idx, numerical, labels in tqdm(test_loader, desc='Evaluating'):
            # 将数据移到指定设备
            color_idx = color_idx.to(device)
            numerical = numerical.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(color_idx, numerical)
            loss = criterion(outputs, labels)

            # 获取预测结果
            _, preds = outputs.max(1)

            # 统计损失
            test_loss += loss.item()

            # 保存预测结果和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算平均测试损失
    test_loss /= len(test_loader)

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # 打印结果
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test F1 Score: {f1:.4f}')

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else None,
                yticklabels=class_names if class_names else None)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    return accuracy, f1, conf_matrix


def main():
    """主函数，用于模型训练和评估"""
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载数据
    # 注意：这里需要替换为实际的数据路径和类别
    data_path = 'models\\training_data.csv'  # 需要替换为实际路径
    try:
        # 根据文件扩展名选择加载方式
        if data_path.endswith('.csv'):
            # 尝试不同的编码格式
            encodings = ['utf-8', 'gbk', 'gb2312', 'iso-8859-1', 'latin1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(data_path, encoding=encoding)
                    print(f"成功使用{encoding}编码加载数据文件")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"尝试使用{encoding}编码读取时出错: {e}")
            else:
                # 如果所有编码都失败，尝试自动检测编码
                try:
                    import chardet
                    with open(data_path, 'rb') as f:
                        result = chardet.detect(f.read())
                    detected_encoding = result['encoding']
                    print(f"检测到文件编码: {detected_encoding}")
                    df = pd.read_csv(data_path, encoding=detected_encoding)
                except Exception as e:
                    print(f"无法读取CSV文件，将创建示例数据: {e}")
                    raise FileNotFoundError("无法读取CSV文件")
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path, engine='openpyxl')
        else:
            print(f"不支持的文件格式: {data_path}，创建示例数据用于演示")
            raise FileNotFoundError("不支持的文件格式")
    except FileNotFoundError:
        print(f"数据文件 {data_path} 不存在或无法读取，创建示例数据用于演示")
        # 创建示例数据用于演示
        np.random.seed(42)
        n_samples = 1000
        colors = ['红色', '蓝色', '绿色', '黄色', '银色', '金色', '铜色']

        df = pd.DataFrame({
            '颜色': np.random.choice(colors, n_samples),
            '密度': np.random.uniform(1.0, 20.0, n_samples),
            '电阻率': np.random.uniform(0.01, 10.0, n_samples),
            '电阻温度系数': np.random.uniform(-0.01, 0.01, n_samples),
            '比热容': np.random.uniform(100, 900, n_samples),
            '熔点': np.random.uniform(200, 3000, n_samples),
            '沸点': np.random.uniform(500, 5000, n_samples),
            '屈服强度': np.random.uniform(50, 1000, n_samples),
            '抗拉强度': np.random.uniform(100, 2000, n_samples),
            '延展率': np.random.uniform(1, 60, n_samples),
            '热膨胀系数': np.random.uniform(1e-6, 3e-5, n_samples),
            '热值': np.random.uniform(0, 1000, n_samples),
            '杨氏模量': np.random.uniform(10, 400, n_samples),
            '硬度': np.random.uniform(10, 1000, n_samples),
            '疲劳强度': np.random.uniform(50, 800, n_samples),
            '冲击韧性': np.random.uniform(5, 300, n_samples),
            'label': np.random.randint(0, 5, n_samples)  # 假设有5个类别
        })

    # 创建数据预处理器
    preprocessor = MaterialDataPreprocessor()

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        df, preprocessor, batch_size=32, train_ratio=0.7, val_ratio=0.15
    )

    # 获取类别数量和颜色词汇表大小
    num_classes = len(df['label'].unique())
    num_colors = preprocessor.get_vocab_size()

    print(f'Number of classes: {num_classes}')
    print(f'Number of colors: {num_colors}')

    # 创建模型
    model = MaterialAnalysisModel(
        num_colors=num_colors,
        color_embed_dim=32,
        num_numerical_features=len(df.columns) - 2,  # 减去颜色和标签列
        hidden_dim=128,
        num_classes=num_classes,
        dropout=0.3
    )

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    # 训练模型
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=30, device=device, patience=5
    )

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))

    # 评估模型
    class_names = [f'Class {i}' for i in range(num_classes)]  # 替换为实际的类别名称
    evaluate_model(model, test_loader, criterion,
                   device=device, class_names=class_names)


if __name__ == '__main__':
    main()
