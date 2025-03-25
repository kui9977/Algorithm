import os
import argparse
import pandas as pd
import torch
import numpy as np
import json
from datetime import datetime

from models.model import MaterialAnalysisModel
from data.preprocessor import MaterialDataPreprocessor, create_dataloaders
from train import train_model, evaluate_model
from predict import MaterialPredictor, save_training_artifacts


def parse_args():
    parser = argparse.ArgumentParser(description='材料识别系统训练与预测流程')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'evaluate'],
                        help='运行模式: train (训练), predict (预测), evaluate (评估)')
    parser.add_argument('--data', type=str, required=True,
                        help='数据文件路径 (CSV格式)')
    parser.add_argument('--model_dir', type=str,
                        default='models', help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--patience', type=int, default=5, help='早停的耐心值')
    parser.add_argument('--predict_input', type=str, help='用于预测的输入文件 (CSV格式)')
    parser.add_argument('--predict_output', type=str, help='预测结果输出文件 (CSV格式)')
    parser.add_argument('--class_names', type=str, help='类别名称文件 (JSON格式)')

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 确保模型目录存在
    os.makedirs(args.model_dir, exist_ok=True)

    # 加载数据 - 支持CSV和Excel格式
    try:
        # 根据文件扩展名选择加载方式
        if args.data.lower().endswith(('.xlsx', '.xls')):
            try:
                # 使用with语句确保文件正确关闭
                with open(args.data, 'rb') as excel_file:
                    if args.data.lower().endswith('.xlsx'):
                        df = pd.read_excel(excel_file, engine='openpyxl')
                    else:
                        df = pd.read_excel(excel_file, engine='xlrd')
                print(f"成功加载Excel数据文件: {args.data}")
            except Exception as e:
                print(f"读取Excel文件失败: {e}")
                return
        elif args.data.endswith('.csv'):
            # 尝试不同的编码格式
            encodings = ['utf-8', 'gbk', 'gb2312', 'iso-8859-1',
                         'latin1', 'utf-16', 'cp936', 'big5']
            success = False

            for encoding in encodings:
                try:
                    print(f"尝试使用 {encoding} 编码读取文件...")
                    df = pd.read_csv(args.data, encoding=encoding)
                    print(f"成功使用 {encoding} 编码加载数据文件")
                    success = True
                    break
                except UnicodeDecodeError as e:
                    print(f"使用 {encoding} 编码失败: {e}")
                    continue
                except Exception as e:
                    print(f"尝试使用 {encoding} 编码读取时出错: {e}")

            if not success:
                # 如果所有编码都失败，尝试使用二进制模式打开并检测编码
                try:
                    import chardet
                    print("尝试使用chardet自动检测文件编码...")
                    with open(args.data, 'rb') as f:
                        raw_data = f.read()
                        result = chardet.detect(raw_data)

                    detected_encoding = result['encoding']
                    confidence = result['confidence']
                    print(
                        f"检测到文件编码: {detected_encoding}，置信度: {confidence:.2f}")

                    if detected_encoding is None:
                        print("无法检测到文件编码，尝试使用binary模式...")
                        # 尝试使用pandas的二进制选项
                        df = pd.read_csv(
                            args.data, encoding='latin1', encoding_errors='replace')
                    else:
                        df = pd.read_csv(args.data, encoding=detected_encoding)

                    print(f"成功使用检测到的编码加载数据文件")
                except Exception as e:
                    print(f"使用自动检测编码仍然失败: {e}")
                    raise ValueError(f"无法读取CSV文件，请检查文件编码或尝试将文件转换为Excel格式: {e}")

        else:
            raise ValueError(f"不支持的文件格式: {args.data}，请使用.csv或.xlsx/.xls格式")

        print(f"成功加载数据文件: {args.data}")
        print(f"数据形状: {df.shape}")
        print(f"数据列: {df.columns.tolist()}")
        print(f"数据预览:\n{df.head()}")
    except Exception as e:
        print(f"加载数据文件时出错: {e}")
        import traceback
        traceback.print_exc()  # 打印详细的错误堆栈
        return

    # 加载类别名称
    class_names = None
    if args.class_names and os.path.exists(args.class_names):
        with open(args.class_names, 'r', encoding='utf-8') as f:
            class_names = json.load(f)
            print(f"加载了 {len(class_names)} 个类别名称")

    # 训练模式
    if args.mode == 'train':
        # 创建预处理器
        preprocessor = MaterialDataPreprocessor()

        # 创建数据加载器
        train_loader, val_loader, test_loader = create_dataloaders(
            df, preprocessor, batch_size=args.batch_size,
            train_ratio=0.7, val_ratio=0.15
        )

        # 获取特征数量和类别数量
        num_colors = preprocessor.get_vocab_size()
        num_classes = len(df['label'].unique())
        num_numerical_features = len(df.columns) - 2  # 减去颜色和标签列

        print(f"颜色数量: {num_colors}")
        print(f"类别数量: {num_classes}")
        print(f"数值特征数量: {num_numerical_features}")

        # 创建模型
        model = MaterialAnalysisModel(
            num_colors=num_colors,
            color_embed_dim=32,
            num_numerical_features=num_numerical_features,
            hidden_dim=128,
            num_classes=num_classes,
            dropout=0.3
        )

        # 定义损失函数和优化器
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )

        # 训练模型
        print("开始训练模型...")
        model = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=args.epochs, device=device, patience=args.patience
        )

        # 保存最终模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_model_path = os.path.join(
            args.model_dir, f"final_model_{timestamp}.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"最终模型已保存到: {final_model_path}")

        # 复制最佳模型到标准位置
        best_model_path = os.path.join(args.model_dir, "best_model.pth")
        import shutil
        shutil.copy("best_model.pth", best_model_path)
        print(f"最佳模型已复制到: {best_model_path}")

        # 评估模型
        print("在测试集上评估模型...")
        accuracy, f1, _ = evaluate_model(
            model, test_loader, criterion, device=device,
            class_names=class_names if class_names else None
        )

        # 创建和保存类别映射
        class_indices = df['label'].unique()
        if class_names:
            class_mapping = {int(idx): class_names[int(
                idx)] for idx in class_indices if int(idx) < len(class_names)}
        else:
            class_mapping = {int(idx): f"类别{idx}" for idx in class_indices}

        # 数值特征列
        numerical_columns = [
            col for col in df.columns if col not in ['颜色', 'label']]

        # 保存训练产物
        save_training_artifacts(
            model, preprocessor, class_mapping, numerical_columns
        )

        print(f"训练完成！最终测试准确率: {accuracy:.4f}, F1分数: {f1:.4f}")

    # 预测模式
    elif args.mode == 'predict':
        if not args.predict_input:
            print("错误: 预测模式需要提供 --predict_input 参数")
            return

        # 加载预测数据 - 支持CSV和Excel格式
        try:
            # 根据文件扩展名选择加载方式
            if args.predict_input.endswith('.csv'):
                predict_df = pd.read_csv(args.predict_input)
            elif args.predict_input.endswith(('.xlsx', '.xls')):
                predict_df = pd.read_excel(args.predict_input)
            else:
                raise ValueError(
                    f"不支持的文件格式: {args.predict_input}，请使用.csv或.xlsx/.xls格式")

            print(f"成功加载预测数据: {args.predict_input}")
            print(f"数据形状: {predict_df.shape}")
        except Exception as e:
            print(f"加载预测数据时出错: {e}")
            return

        # 创建预测器
        try:
            model_path = os.path.join(args.model_dir, "best_model.pth")
            predictor = MaterialPredictor(
                model_path=model_path,
                preprocessor_path=os.path.join(
                    args.model_dir, "preprocessor.pkl"),
                metadata_path=os.path.join(args.model_dir, "metadata.json"),
                device=device
            )
        except Exception as e:
            print(f"创建预测器时出错: {e}")
            return

        # 进行批量预测
        print("进行预测...")
        pred_classes, pred_probs, class_names = predictor.predict_batch(
            predict_df)

        # 将预测结果添加到数据框
        predict_df['predicted_class'] = pred_classes
        predict_df['confidence'] = pred_probs
        predict_df['class_name'] = class_names

        # 保存预测结果
        output_path = args.predict_output or os.path.join(
            os.path.dirname(args.predict_input),
            f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        predict_df.to_csv(output_path, index=False)
        print(f"预测结果已保存到: {output_path}")

    # 评估模式
    elif args.mode == 'evaluate':
        # 创建预处理器
        preprocessor = MaterialDataPreprocessor()

        # 创建数据加载器
        _, _, test_loader = create_dataloaders(
            df, preprocessor, batch_size=args.batch_size,
            train_ratio=0.0, val_ratio=0.0  # 全部作为测试集
        )

        # 获取特征数量和类别数量
        num_colors = preprocessor.get_vocab_size()
        num_classes = len(df['label'].unique())
        num_numerical_features = len(df.columns) - 2  # 减去颜色和标签列

        # 创建模型
        model = MaterialAnalysisModel(
            num_colors=num_colors,
            color_embed_dim=32,
            num_numerical_features=num_numerical_features,
            hidden_dim=128,
            num_classes=num_classes,
            dropout=0.0  # 评估时不使用dropout
        )

        # 加载模型权重
        model_path = os.path.join(args.model_dir, "best_model.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        # 定义损失函数
        criterion = torch.nn.CrossEntropyLoss()

        # 评估模型
        print(f"使用模型 {model_path} 在数据集上进行评估...")
        accuracy, f1, _ = evaluate_model(
            model, test_loader, criterion, device=device,
            class_names=class_names if class_names else None
        )

        print(f"评估完成！准确率: {accuracy:.4f}, F1分数: {f1:.4f}")


if __name__ == '__main__':
    main()
