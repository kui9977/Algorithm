"""
金属材料多模态识别系统 API 客户端示例
演示如何从后端调用API
"""

import requests
import json
import base64
import os
import io
from PIL import Image

class MetalAPIClient:
    """金属材料识别API客户端"""
    
    def __init__(self, base_url="http://localhost:5000"):
        """初始化客户端"""
        self.base_url = base_url
    
    def get_known_colors(self):
        """获取已知的颜色列表"""
        try:
            response = requests.get(f"{self.base_url}/api/known_colors")
            if response.status_code == 200:
                return response.json()['colors']
            else:
                print(f"请求失败: {response.status_code}")
                print(response.text)
                return []
        except Exception as e:
            print(f"请求出错: {e}")
            return []
    
    def predict_material(self, color, **features):
        """
        预测材料类型
        
        参数:
        - color: 材料颜色
        - features: 其他特征，如density(密度)等
        
        返回:
        - 预测结果字典
        """
        try:
            # 准备请求数据
            data = {'color': color}
            data.update(features)
            
            # 发送请求
            response = requests.post(
                f"{self.base_url}/api/predict",
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                print(f"预测请求失败: {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"预测请求出错: {e}")
            return None
    
    def save_result_image(self, result, save_path):
        """
        保存预测结果图像
        
        参数:
        - result: predict_material返回的结果
        - save_path: 图像保存路径
        """
        if result and 'result_image' in result:
            try:
                # 解码Base64图像
                img_data = base64.b64decode(result['result_image'])
                img = Image.open(io.BytesIO(img_data))
                img.save(save_path)
                print(f"结果图像已保存至: {save_path}")
                return True
            except Exception as e:
                print(f"保存图像失败: {e}")
                return False
        else:
            print("结果中不包含图像数据")
            return False

def example_usage():
    """API调用示例"""
    client = MetalAPIClient()
    
    # 1. 获取已知颜色列表
    print("获取已知颜色列表...")
    colors = client.get_known_colors()
    print(f"可用颜色: {colors[:10]}...")
    print()
    
    # 2. 预测示例 - 完整特征
    print("示例1: 使用完整特征预测 (铜)")
    result1 = client.predict_material(
        color="紫红色",
        density=8.96,
        resistivity=1.678,
        specific_heat=0.39,
        melting_point=1083,
        boiling_point=2567,
        yield_strength=220,
        tensile_strength=240,
        elongation=35,
        thermal_expansion=1.485e-05,
        heat_value=24160,
        youngs_modulus=130,
        hardness=40,
        fatigue_strength=139.3,
        impact_toughness=42.5
    )
    
    if result1:
        print("预测结果:")
        for i, item in enumerate(result1['results']):
            print(f"Top {i+1}: {item['material']} (概率: {item['probability']:.4f})")
        
        # 保存结果图像
        client.save_result_image(result1, "prediction_complete.png")
    print()
    
    # 3. 预测示例 - 部分特征
    print("示例2: 使用部分特征预测 (仅颜色和密度)")
    result2 = client.predict_material(
        color="紫红色",
        density=8.96
    )
    
    if result2:
        print("预测结果:")
        for i, item in enumerate(result2['results']):
            print(f"Top {i+1}: {item['material']} (概率: {item['probability']:.4f})")
        
        # 保存结果图像
        client.save_result_image(result2, "prediction_partial.png")

if __name__ == "__main__":
    example_usage()
