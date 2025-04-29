#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import requests
import base64
import matplotlib.pyplot as plt
import numpy as np
import io
import os
from PIL import Image

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='金属材料多模态识别系统API客户端示例')
    parser.add_argument('--url', type=str, default='http://localhost:5000',
                        help='API服务地址 (默认: http://localhost:5000)')
    return parser.parse_args()

def check_health(base_url):
    """健康检查"""
    url = f"{base_url}/api/health"
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查HTTP错误
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"健康检查失败: {e}")
        if hasattr(e.response, 'text'):
            print(f"错误响应: {e.response.text}")
        return None

def get_known_colors(base_url):
    """获取已知颜色列表"""
    url = f"{base_url}/api/known_colors"
    try:
        response = requests.get(url)
        response.raise_for_status()
        result = response.json()
        # 兼容处理两种不同的返回格式
        if 'colors' in result:
            return result.get('colors', [])
        else:
            print(f"API返回格式不符合预期: {result}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"获取颜色列表失败: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"错误响应: {e.response.text}")
        return []

def predict_material(base_url, features):
    """预测材料类型"""
    url = f"{base_url}/api/predict"
    try:
        response = requests.post(url, json=features)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"预测失败: {e}")
        if hasattr(e.response, 'text'):
            print(f"错误响应: {e.response.text}")
        return None

def display_image_from_base64(base64_str, save_path=None):
    """从base64字符串显示图像，并可选择保存到文件"""
    try:
        # 解码base64字符串
        img_data = base64.b64decode(base64_str)
        
        # 从二进制数据创建图像
        img = Image.open(io.BytesIO(img_data))
        
        # 如果提供了保存路径，则保存图像
        if save_path:
            # 确保保存目录存在
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            img.save(save_path)
            print(f"图像已保存至: {save_path}")
        
        # 显示图像
        plt.figure(figsize=(10, 6))
        plt.imshow(np.array(img))
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"处理图像失败: {e}")

def download_prediction_image(base_url, image_url, save_path):
    """直接从API下载预测结果图像"""
    try:
        # 组合完整的URL
        full_url = f"{base_url}{image_url}" if image_url.startswith('/') else f"{base_url}/{image_url}"
        
        # 发送请求获取图像
        response = requests.get(full_url, stream=True)
        response.raise_for_status()
        
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 保存图像到文件
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"预测结果图像已下载至: {save_path}")
        return True
    except Exception as e:
        print(f"下载图像失败: {e}")
        return False

def main():
    args = parse_args()
    base_url = args.url
    
    # 查看API根路径
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            print("API服务信息:")
            print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except:
        print("无法访问API根路径")
    
    # 健康检查
    print("\n执行健康检查...")
    health_result = check_health(base_url)
    if health_result:
        print(f"健康状态: {health_result.get('status')}")
        print(f"消息: {health_result.get('message')}")
    else:
        print("无法连接到API服务，请确保服务已启动")
        return
    
    # 示例预测
    print("\n示例预测 - 铜材料:")
    features = {
        "color": "紫红色",
        "density": 8.96,
        "resistivity": 1.678e-8,
        "specific_heat": 0.39,
        "melting_point": 1083,
        "boiling_point": 2567
    }
    
    print(f"输入特征: {json.dumps(features, ensure_ascii=False, indent=2)}")
    result = predict_material(base_url, features)
    
    if result and result.get('success'):
        print("\n预测结果:")
        for i, material in enumerate(result.get('results', []), 1):
            print(f"  {i}. {material['material']}: {material['probability']:.4f}")
        
        # 创建保存目录
        save_dir = 'D:\\Projects\\Python_projects\\dpl\\Algorithm\\prediction_results'
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        top_material = result.get('results', [{}])[0].get('material', 'unknown')
        top_material = top_material.replace('/', '_').replace('\\', '_')  # 替换可能导致路径问题的字符
        save_path = os.path.join(save_dir, f"{timestamp}_{top_material}.png")
        
        # 1. 显示并保存base64编码的图像
        if 'result_image' in result:
            print("\n显示并保存预测结果图表...")
            display_image_from_base64(result['result_image'], save_path)
        
        # 2. 如果有图像URL，也直接下载图像
        if 'image_url' in result:
            download_path = os.path.join(save_dir, f"{timestamp}_{top_material}_direct.png")
            download_prediction_image(base_url, result['image_url'], download_path)
    else:
        print(f"预测失败: {result.get('error') if result else '未知错误'}")
    
    # 交互式预测
    while True:
        print("\n是否要进行新的预测? (y/n)")
        choice = input().strip().lower()
        if choice != 'y':
            break
        
        # 获取已知颜色列表
        colors = get_known_colors(base_url)
        if not colors:
            print("无法获取材料颜色列表")
            continue
            
        # 获取用户输入
        print(f"请选择材料颜色 (1-{len(colors)}):")
        for i, color in enumerate(colors, 1):
            print(f"  {i}. {color}")
        
        try:
            color_idx = int(input().strip()) - 1
            if color_idx < 0 or color_idx >= len(colors):
                print("无效的颜色选择")
                continue
            
            selected_color = colors[color_idx]
            
            # 创建特征字典
            features = {"color": selected_color}
            
            # 可选数值特征
            print("\n请输入已知的数值特征 (不知道的请留空):")
            
            density = input("密度(g/cm³): ").strip()
            if density:
                features["density"] = float(density)
            
            resistivity = input("电阻率: ").strip()
            if resistivity:
                features["resistivity"] = float(resistivity)
            
            specific_heat = input("比热容: ").strip()
            if specific_heat:
                features["specific_heat"] = float(specific_heat)
            
            melting_point = input("熔点: ").strip()
            if melting_point:
                features["melting_point"] = float(melting_point)
            
            # 执行预测
            print("\n执行预测...")
            result = predict_material(base_url, features)
            
            if result and result.get('success'):
                print("\n预测结果:")
                for i, material in enumerate(result.get('results', []), 1):
                    print(f"  {i}. {material['material']}: {material['probability']:.4f}")
                
                # 创建保存目录
                save_dir = 'D:\\Projects\\Python_projects\\dpl\\Algorithm\\prediction_results'
                os.makedirs(save_dir, exist_ok=True)
                
                # 生成文件名
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                top_material = result.get('results', [{}])[0].get('material', 'unknown')
                top_material = top_material.replace('/', '_').replace('\\', '_')
                save_path = os.path.join(save_dir, f"{timestamp}_{top_material}.png")
                
                # 显示并保存图像
                if 'result_image' in result:
                    print("\n显示并保存预测结果图表...")
                    display_image_from_base64(result['result_image'], save_path)
                
                # 如果有图像URL，也直接下载图像
                if 'image_url' in result:
                    download_path = os.path.join(save_dir, f"{timestamp}_{top_material}_direct.png")
                    download_prediction_image(base_url, result['image_url'], download_path)
            else:
                print(f"预测失败: {result.get('error') if result else '未知错误'}")
                
        except ValueError:
            print("输入错误，请确保输入正确的数值")
        except Exception as e:
            print(f"发生错误: {e}")

if __name__ == "__main__":
    main()
