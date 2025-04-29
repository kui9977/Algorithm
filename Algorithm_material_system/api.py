#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import base64
import io
from flask import Flask, request, jsonify, send_file, Blueprint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
import logging
import sys

# 设置项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# 确保模型目录存在
os.makedirs(MODELS_DIR, exist_ok=True)

# 将项目根目录添加到系统路径
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# 导入项目模块
from model import MetalClassifier, load_model
from data_preprocessing import load_preprocessor

# 初始化Flask应用
app = Flask(__name__)
logger = logging.getLogger(__name__)

# 创建API蓝图
api_bp = Blueprint('api', __name__, url_prefix='/api')

# 加载预处理器和模型
def load_resources():
    try:
        # 加载预处理器
        preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
        preprocessor = load_preprocessor(preprocessor_path)
        
        # 加载材料名称
        material_names_path = os.path.join(MODELS_DIR, 'material_names.pkl')
        with open(material_names_path, 'rb') as f:
            material_names = pickle.load(f)
        
        # 加载类别索引
        class_indices_path = os.path.join(MODELS_DIR, 'class_indices.pkl')
        with open(class_indices_path, 'rb') as f:
            class_indices = pickle.load(f)
        
        # 确定类别数和输入维度
        input_dim = 15  # 1个颜色特征 + 14个数值特征
        hidden_dims = [256, 128, 64]
        
        # 先尝试从保存的模型获取实际类别数
        try:
            model_path = os.path.join(MODELS_DIR, 'metal_classifier.pth')
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            if 'model.12.weight' in state_dict:
                num_classes = state_dict['model.12.weight'].size(0)
                logger.info(f"从模型权重检测到 {num_classes} 个输出类别")
            else:
                # 如果无法从模型中直接检测，使用材料名称数量
                num_classes = len(material_names)
                logger.info(f"未能从模型权重检测到类别数，使用材料名称数量: {num_classes}")
        except Exception as e:
            # 如果无法加载模型进行检测，使用最大类别数
            logger.warning(f"从模型中检测类别数失败: {e}")
            num_classes = max(len(material_names), len(class_indices), 104)
            logger.info(f"使用估计的类别数: {num_classes}")
        
        # 使用load_model函数加载模型(自动处理类别数)
        model_path = os.path.join(MODELS_DIR, 'metal_classifier.pth')
        model = load_model(input_dim, hidden_dims, num_classes, model_path)
        model.eval()  # 确保模型处于评估模式
        
        return {
            'preprocessor': preprocessor,
            'material_names': material_names,
            'class_indices': class_indices,
            'model': model,
            'num_classes': num_classes  # 记录实际使用的类别数
        }
    except Exception as e:
        logger.error(f"加载资源失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# 尝试加载资源
resources = load_resources()

# 健康检查接口
@api_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查API接口"""
    return jsonify({
        'status': 'ok',
        'message': '金属材料多模态识别系统API正常运行中'
    })

# 获取已知颜色列表接口
@api_bp.route('/known_colors', methods=['GET'])
def get_known_colors():
    """获取系统已知的颜色列表"""
    if resources is None:
        return jsonify({'success': False, 'error': '系统资源未正确加载', 'colors': []}), 500
    
    try:
        # 从预处理器中获取颜色列表
        color_encoder = resources['preprocessor']['color']
        colors = color_encoder.classes_.tolist()
        
        return jsonify({'success': True, 'colors': colors})
    except Exception as e:
        logger.error(f"获取颜色列表失败: {e}")
        return jsonify({'success': False, 'error': f'获取颜色列表失败: {str(e)}', 'colors': []}), 500

# 预测接口
@api_bp.route('/predict', methods=['POST'])
def predict():
    """根据输入特征预测材料类型"""
    if resources is None:
        return jsonify({'error': '系统资源未正确加载'}), 500
    
    # 获取请求数据
    data = request.json
    if not data:
        return jsonify({'error': '未提供任何特征数据'}), 400
    
    # 验证必需参数
    if 'color' not in data:
        return jsonify({'error': '颜色特征必须提供'}), 400
    
    try:
        # 准备预测数据
        sample_dict = {
            '颜色': data.get('color')
        }
        
        # 将其他特征添加到sample_dict
        feature_mapping = {
            'density': '密度(g/cm3)',
            'resistivity': '电阻率',
            'specific_heat': '比热容',
            'melting_point': '熔点',
            'boiling_point': '沸点',
            'yield_strength': '屈服强度',
            'tensile_strength': '抗拉强度',
            'elongation': '延展率',
            'thermal_expansion': '热膨胀系数',
            'heat_value': '热值(J/kg)',
            'youngs_modulus': '杨氏模量GPa',
            'hardness': '硬度',
            'fatigue_strength': '疲劳强度',
            'impact_toughness': '冲击韧性J/cm2'
        }
        
        # 将API参数映射到模型所需参数格式
        for api_key, model_key in feature_mapping.items():
            if api_key in data and data[api_key] is not None:
                sample_dict[model_key] = data[api_key]
        
        # 预处理输入特征
        preprocessor = resources['preprocessor']
        model = resources['model']
        material_names = resources['material_names']
        class_indices = resources['class_indices']
        
        # 使用预处理器处理样本
        features = preprocessor.preprocess_single_sample(sample_dict)
        
        # 进行预测
        with torch.no_grad():
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # 获取前k个预测结果
        k = min(5, len(probabilities))
        top_indices = np.argsort(probabilities)[::-1][:k]
        top_probs = probabilities[top_indices]
        
        # 创建索引到名称的反向映射
        idx_to_name = {v: k for k, v in class_indices.items()}
        
        # 准备返回结果
        results = []
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            name = idx_to_name.get(idx, f"未知类别 {idx}")
            results.append({
                'material': name,
                'probability': float(prob),
                'index': int(idx)
            })
        
        # 生成预测结果可视化图表
        plt.figure(figsize=(10, 6))
        materials = [result['material'] for result in results]
        probs = [result['probability'] for result in results]
        
        # 画条形图
        bars = plt.barh(materials[::-1], probs[::-1], color='cornflowerblue')
        plt.xlabel('概率值')
        plt.ylabel('材料名称')
        plt.title('材料分类预测结果')
        plt.xlim(0, 1)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        # 在条形上添加数值标签
        for bar, prob in zip(bars, probs[::-1]):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                     f'{prob:.4f}', va='center')
        
        # 将图表转换为base64字符串
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return jsonify({
            'success': True,
            'results': results,
            'result_image': img_base64
        })
    
    except Exception as e:
        logger.error(f"预测过程出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'预测过程出错: {str(e)}'}), 500

# 根路径接口
@app.route('/', methods=['GET'])
def index():
    """根路径重定向到健康检查页面"""
    return jsonify({
        'message': '金属材料多模态识别系统API服务',
        'health_endpoint': '/api/health',
        'colors_endpoint': '/api/known_colors',
        'predict_endpoint': '/api/predict'
    })

# 注册蓝图
app.register_blueprint(api_bp)

# 404错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '请求的资源不存在'}), 404

# 500错误处理
@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': '服务器内部错误'}), 500

if __name__ == '__main__':
    # 直接运行此文件时，以调试模式启动
    app.run(debug=True, port=5000)
