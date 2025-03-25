import torch
import pandas as pd
import numpy as np
import joblib
import os
import json
import logging

from models.model import MaterialAnalysisModel
from data.preprocessor import MaterialDataPreprocessor

# 配置日志
logger = logging.getLogger("material_predictor")


class MaterialPredictor:
    def __init__(self, model_path='best_model.pth', preprocessor_path='preprocessor.pkl',
                 metadata_path='metadata.json', device=None):
        """
        初始化材料识别预测器

        参数:
        model_path: 模型权重文件路径
        preprocessor_path: 预处理器文件路径
        metadata_path: 元数据文件路径(包含类别映射等信息)
        device: 计算设备，如果为None则自动选择
        """
        # 设置设备
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        logger.info(f"使用设备: {self.device}")

        # 加载预处理器
        if os.path.exists(preprocessor_path):
            self.preprocessor = joblib.load(preprocessor_path)
            logger.info(f"已加载预处理器: {preprocessor_path}")
        else:
            raise FileNotFoundError(f"预处理器文件不存在: {preprocessor_path}")

        # 加载元数据
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"已加载元数据: {metadata_path}")
        else:
            raise FileNotFoundError(f"元数据文件不存在: {metadata_path}")

        # 创建模型
        num_colors = self.preprocessor.get_vocab_size()
        num_classes = len(self.metadata['class_mapping'])
        num_numerical_features = len(self.metadata['numerical_columns'])

        self.model = MaterialAnalysisModel(
            num_colors=num_colors,
            color_embed_dim=32,
            num_numerical_features=num_numerical_features,
            hidden_dim=128,
            num_classes=num_classes,
            dropout=0.0  # 推理时不使用dropout
        )

        # 加载模型权重
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(
                model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"已加载模型权重: {model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 获取类别映射
        self.idx_to_class = {
            int(k): v for k, v in self.metadata['class_mapping'].items()}

    def predict(self, sample_data):
        """
        对单个样本进行预测

        参数:
        sample_data: 字典或DataFrame，包含材料的属性

        返回:
        预测类别、预测概率、预测类别名称
        """
        # 转换为DataFrame
        if isinstance(sample_data, dict):
            df = pd.DataFrame([sample_data])
        else:
            df = sample_data.copy()

        # 预处理数据
        color_indices, numerical_features, _ = self.preprocessor.transform(df)

        # 转换为张量
        color_tensor = torch.tensor(
            color_indices, dtype=torch.long).to(self.device)
        numerical_tensor = torch.tensor(
            numerical_features, dtype=torch.float32).to(self.device)

        # 模型预测
        with torch.no_grad():
            outputs = self.model(color_tensor, numerical_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            # 获取预测类别和概率
            probs, preds = torch.max(probabilities, dim=1)

            # 转换为NumPy数组
            pred_class = preds.cpu().numpy()[0]
            pred_prob = probs.cpu().numpy()[0]

            # 获取类别名称
            class_name = self.idx_to_class.get(
                pred_class, f"未知类别({pred_class})")

            return pred_class, pred_prob, class_name

    def predict_batch(self, batch_data):
        """
        对一批样本进行预测

        参数:
        batch_data: DataFrame，包含多个材料样本

        返回:
        预测类别列表、预测概率列表、预测类别名称列表
        """
        # 预处理数据
        color_indices, numerical_features, _ = self.preprocessor.transform(
            batch_data)

        # 转换为张量
        color_tensor = torch.tensor(
            color_indices, dtype=torch.long).to(self.device)
        numerical_tensor = torch.tensor(
            numerical_features, dtype=torch.float32).to(self.device)

        # 模型预测
        with torch.no_grad():
            outputs = self.model(color_tensor, numerical_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            # 获取预测类别和概率
            probs, preds = torch.max(probabilities, dim=1)

            # 转换为NumPy数组
            pred_classes = preds.cpu().numpy()
            pred_probs = probs.cpu().numpy()

            # 获取类别名称
            class_names = [self.idx_to_class.get(
                cls, f"未知类别({cls})") for cls in pred_classes]

            return pred_classes, pred_probs, class_names

    def explain_prediction(self, sample_data):
        """
        解释预测结果，提供材料的关键特性

        参数:
        sample_data: 字典或DataFrame，包含材料的属性

        返回:
        解释信息字典
        """
        # 预测类别
        pred_class, pred_prob, class_name = self.predict(sample_data)

        # 获取类别的典型特性
        typical_properties = self.metadata.get(
            'class_properties', {}).get(str(pred_class), {})

        # 获取材料可能的应用场景
        applications = self.metadata.get(
            'class_applications', {}).get(str(pred_class), [])

        # 构建解释信息
        explanation = {
            'predicted_class': int(pred_class),
            'class_name': class_name,
            'confidence': float(pred_prob),
            'typical_properties': typical_properties,
            'possible_applications': applications
        }

        return explanation


def save_training_artifacts(model, preprocessor, class_mapping, numerical_columns,
                            class_properties=None, class_applications=None, model_dir='.'):
    """
    保存训练产物，用于后续预测

    参数:
    model: 训练好的模型
    preprocessor: 预处理器
    class_mapping: 类别映射，从ID到类别名称
    numerical_columns: 数值特征列名列表
    class_properties: 每个类别的典型属性
    class_applications: 每个类别的应用场景
    model_dir: 模型保存目录
    """
    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)

    # 保存预处理器
    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"预处理器已保存到: {preprocessor_path}")

    # 准备元数据
    metadata = {
        'class_mapping': class_mapping,
        'numerical_columns': numerical_columns,
        'model_info': {
            'num_classes': len(class_mapping),
            'num_features': len(numerical_columns),
            'num_colors': preprocessor.get_vocab_size()
        }
    }

    # 添加类别属性和应用场景（如果提供）
    if class_properties:
        metadata['class_properties'] = class_properties

    if class_applications:
        metadata['class_applications'] = class_applications

    # 保存元数据
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    logger.info(f"元数据已保存到: {metadata_path}")

    logger.info("训练产物已保存，可用于预测。")


def example_usage():
    """示例用法"""
    # 1. 创建预测器
    try:
        predictor = MaterialPredictor(
            model_path='best_model.pth',
            preprocessor_path='preprocessor.pkl',
            metadata_path='metadata.json'
        )

        # 2. 准备样本数据
        sample = {
            '颜色': '银色',
            '密度': 7.85,
            '电阻率': 9.71e-8,
            '电阻温度系数': 0.0065,
            '比热容': 450,
            '熔点': 1538,
            '沸点': 2862,
            '屈服强度': 250,
            '抗拉强度': 420,
            '延展率': 22,
            '热膨胀系数': 1.2e-5,
            '热值': 0,
            '杨氏模量': 210,
            '硬度': 150,
            '疲劳强度': 170,
            '冲击韧性': 30
        }

        # 3. 进行预测
        pred_class, pred_prob, class_name = predictor.predict(sample)
        print(f"预测类别: {class_name} (ID: {pred_class})")
        print(f"预测概率: {pred_prob:.4f}")

        # 4. 获取解释信息
        explanation = predictor.explain_prediction(sample)
        print("\n预测解释:")
        print(f"材料类别: {explanation['class_name']}")
        print(f"置信度: {explanation['confidence']:.2f}")

        if explanation.get('typical_properties'):
            print("\n典型属性:")
            for prop, value in explanation['typical_properties'].items():
                print(f"- {prop}: {value}")

        if explanation.get('possible_applications'):
            print("\n可能的应用场景:")
            for app in explanation['possible_applications']:
                print(f"- {app}")

    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先训练模型并保存相关文件。")


if __name__ == '__main__':
    example_usage()
