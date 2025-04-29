"""
金属材料多模态识别系统 API 模块
提供RESTful API接口，方便后端调用算法
"""

import os
import json
import torch
import numpy as np
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
import time
from flask import Flask, request, jsonify
from flask_cors import CORS

# 导入自定义模块
from model import load_model
from data_preprocessing import load_preprocessor
from predict import set_chinese_font, plot_prediction_results

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量存储加载的模型和预处理器
MODEL = None
PREPROCESSOR = None
MATERIAL_NAMES = None
CLASS_INDICES = None

def load_resources():
    """加载模型资源"""
    global MODEL, PREPROCESSOR, MATERIAL_NAMES, CLASS_INDICES
    
    try:
        # 加载预处理器
        PREPROCESSOR = load_preprocessor()
        
        # 加载材料名称列表
        import pickle
        material_names_path = 'models/material_names.pkl'
        class_indices_path = 'models/class_indices.pkl'
        
        if os.path.exists(material_names_path):
            with open(material_names_path, 'rb') as f:
                MATERIAL_NAMES = pickle.load(f)
        else:
            raise FileNotFoundError("找不到材料名称列表文件")
        
        # 加载类别索引映射
        if os.path.exists(class_indices_path):
            with open(class_indices_path, 'rb') as f:
                CLASS_INDICES = pickle.load(f)
        
        # 加载模型
        input_dim = 15  # 1个颜色特征 + 14个数值特征
        hidden_dims = [256, 128, 64]
        num_classes = 104  # 数据集中的类别数量
        MODEL = load_model(input_dim, hidden_dims, num_classes)
        
        return True
    except Exception as e:
        print(f"加载资源出错: {e}")
        return False

def predict_material(features):
    """预测材料类型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL.to(device)
    features = features.to(device)
    
    # 进行预测
    MODEL.eval()
    with torch.no_grad():
        outputs = MODEL(features)
        probabilities = torch.softmax(outputs, dim=1)
    
    # 获取预测结果
    probabilities = probabilities.cpu().numpy()[0]
    
    # 获取前k个预测结果
    k = min(5, len(probabilities))
    top_k_indices = np.argsort(probabilities)[::-1][:k]
    top_k_probs = probabilities[top_k_indices]
    
    # 使用类别索引映射来获取材料名称
    top_k_preds = []
    if CLASS_INDICES:
        # 创建索引到名称的反向映射
        idx_to_name = {v: k for k, v in CLASS_INDICES.items()}
        for idx, prob in zip(top_k_indices, top_k_probs):
            name = idx_to_name.get(idx, f"未知类别 {idx}")
            top_k_preds.append((idx, prob, name))
    else:
        # 如果没有提供类别索引映射，使用材料名称列表
        for idx, prob in zip(top_k_indices, top_k_probs):
            if idx < len(MATERIAL_NAMES):
                name = MATERIAL_NAMES[idx]
            else:
                name = f"未知类别 {idx}"
            top_k_preds.append((idx, prob, name))
    
    return top_k_preds

def generate_result_image(top_k_preds):
    """生成预测结果图像，并返回Base64编码的图像数据"""
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
    
    # 将图像保存到内存中的BytesIO对象
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 将图像转换为Base64编码
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    return img_base64

# API路由
@app.route('/api/known_colors', methods=['GET'])
def get_known_colors():
    """获取已知的颜色列表"""
    if PREPROCESSOR is None:
        if not load_resources():
            return jsonify({'error': '资源加载失败'}), 500
    
    try:
        colors = PREPROCESSOR.get_known_colors()
        return jsonify({'colors': colors})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """预测材料类型"""
    if PREPROCESSOR is None or MODEL is None:
        if not load_resources():
            return jsonify({'error': '资源加载失败'}), 500
    
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({'error': '请求数据为空'}), 400
        
        # 验证必需的颜色字段
        if 'color' not in data or not data['color']:
            return jsonify({'error': '颜色是必须的参数'}), 400
        
        # 创建样本数据字典
        sample = {'颜色': data['color']}
        
        # 添加数值特征（如果提供）
        numeric_features = {
            '密度(g/cm3)': 'density',
            '电阻率': 'resistivity',
            '比热容': 'specific_heat',
            '熔点': 'melting_point',
            '沸点': 'boiling_point',
            '屈服强度': 'yield_strength',
            '抗拉强度': 'tensile_strength',
            '延展率': 'elongation',
            '热膨胀系数': 'thermal_expansion',
            '热值(J/kg)': 'heat_value',
            '杨氏模量GPa': 'youngs_modulus',
            '硬度': 'hardness',
            '疲劳强度': 'fatigue_strength',
            '冲击韧性J/cm2': 'impact_toughness'
        }
        
        for cn_key, en_key in numeric_features.items():
            if en_key in data and data[en_key] is not None:
                try:
                    sample[cn_key] = float(data[en_key])
                except (ValueError, TypeError):
                    # 忽略无效的数值，使用默认值
                    pass
        
        # 预处理输入样本
        features = PREPROCESSOR.preprocess_single_sample(sample)
        
        # 进行预测
        top_k_preds = predict_material(features)
        
        # 生成结果图像的Base64编码
        result_image = generate_result_image(top_k_preds)
        
        # 准备响应数据
        results = []
        for idx, prob, name in top_k_preds:
            results.append({
                'material': name,
                'probability': float(prob),
                'index': int(idx)
            })
        
        response = {
            'success': True,
            'results': results,
            'result_image': result_image
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({'status': 'ok', 'message': '金属材料多模态识别系统API正常运行中'})

# 启动时预加载资源
@app.before_first_request
def initialize():
    """启动时预加载资源"""
    load_resources()

if __name__ == '__main__':
    # 加载资源
    load_resources()
    
    # 启动服务
    app.run(host='0.0.0.0', port=5000, debug=False)
