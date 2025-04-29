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

# 设置项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# 确保模型目录存在
os.makedirs(MODELS_DIR, exist_ok=True)

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
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        # 加载材料名称
        material_names_path = os.path.join(MODELS_DIR, 'material_names.pkl')
        with open(material_names_path, 'rb') as f:
            material_names = pickle.load(f)
        
        # 加载类别索引
        class_indices_path = os.path.join(MODELS_DIR, 'class_indices.pkl')
        with open(class_indices_path, 'rb') as f:
            class_indices = pickle.load(f)
        
        # 加载模型
        model_path = os.path.join(MODELS_DIR, 'metal_classifier.pth')
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        
        return {
            'preprocessor': preprocessor,
            'material_names': material_names,
            'class_indices': class_indices,
            'model': model
        }
    except Exception as e:
        logger.error(f"加载资源失败: {e}")
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
        features = {
            'color': data.get('color'),
            'density': data.get('density'),
            'resistivity': data.get('resistivity'),
            'specific_heat': data.get('specific_heat'),
            'melting_point': data.get('melting_point'),
            'boiling_point': data.get('boiling_point'),
            'yield_strength': data.get('yield_strength'),
            'tensile_strength': data.get('tensile_strength'),
            'elongation': data.get('elongation'),
            'thermal_expansion': data.get('thermal_expansion'),
            'heat_value': data.get('heat_value'),
            'youngs_modulus': data.get('youngs_modulus'),
            'hardness': data.get('hardness'),
            'fatigue_strength': data.get('fatigue_strength'),
            'impact_toughness': data.get('impact_toughness')
        }
        
        # 预处理输入特征
        preprocessor = resources['preprocessor']
        model = resources['model']
        material_names = resources['material_names']
        
        # 转换颜色特征
        color_encoded = preprocessor['color'].transform([features['color']])[0]
        
        # 准备数值特征
        numeric_features = [
            features.get('density'),
            features.get('resistivity'),
            features.get('specific_heat'),
            features.get('melting_point'),
            features.get('boiling_point'),
            features.get('yield_strength'),
            features.get('tensile_strength'),
            features.get('elongation'),
            features.get('thermal_expansion'),
            features.get('heat_value'),
            features.get('youngs_modulus'),
            features.get('hardness'),
            features.get('fatigue_strength'),
            features.get('impact_toughness')
        ]
        
        # 将数值特征转换为numpy数组
        numeric_features = np.array([numeric_features])
        
        # 处理数值特征中的缺失值并进行标准化
        numeric_features_processed = preprocessor['numeric'].transform(numeric_features)
        
        # 合并特征
        combined_features = np.column_stack((color_encoded, numeric_features_processed.flatten()))
        combined_features = torch.FloatTensor(combined_features)
        
        # 预测
        with torch.no_grad():
            output = model(combined_features)
            probabilities = torch.nn.functional.softmax(output, dim=1).numpy()[0]
        
        # 获取前几个最可能的预测结果
        top_k = 5
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_probs = probabilities[top_indices]
        
        # 准备返回结果
        results = []
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
            results.append({
                'material': material_names[idx],
                'probability': float(prob),
                'index': int(idx)
            })
        
        # 生成预测结果可视化图表
        plt.figure(figsize=(10, 6))
        plt.bar(
            [result['material'] for result in results[:5]], 
            [result['probability'] for result in results[:5]]
        )
        plt.xlabel('材料类型')
        plt.ylabel('概率')
        plt.title('材料预测结果概率分布')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 将图表转换为base64字符串
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
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
