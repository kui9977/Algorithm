# 金属材料多模态识别系统

本项目实现了一个基于机器学习的金属材料识别系统，利用多模态深度学习模型，根据材料的物理和化学特性（如颜色、密度、电阻率等）预测材料的种类。系统支持部分特征输入，提高了实用性和灵活性。

## 项目结构

- `complete_materials_train.csv`: 训练数据集，包含各种金属材料及其特性
- `complete_materials_val.csv`: 验证数据集，用于评估模型性能
- `data_preprocessing.py`: 数据预处理模块，处理分类特征和数值特征，支持缺失值处理
- `model.py`: 模型定义模块，包含神经网络结构
- `train_evaluate.py`: 模型训练和评估模块，含可视化功能
- `predict.py`: 使用训练好的模型进行预测，支持部分特征输入
- `save_material_names.py`: 提取并保存材料名称列表
- `run_pipeline.py`: 运行整个算法流程的主脚本
- `api.py`: API服务模块，提供RESTful API接口
- `api_service.py`: API服务启动和管理脚本
- `api_client.py`: API客户端示例代码
- `requirements.txt`: 项目依赖列表
- `models/`: 目录，存放训练好的模型和预处理器
  - `metal_classifier.pth`: 训练好的神经网络模型
  - `preprocessor.pkl`: 保存的预处理器
  - `material_names.pkl`: 材料名称列表
  - `class_indices.pkl`: 类别索引映射
- `learning_curves.png`: 训练过程中的学习曲线图表
- `confusion_matrix.png`: 混淆矩阵可视化图表

## 数据集说明

数据集包含100多种金属材料的特征，每种材料包括：

1. **分类特征**:
   - 颜色（如：银白色、金黄色、黑色等）

2. **数值特征**:
   - 密度(g/cm³)
   - 电阻率
   - 比热容
   - 熔点
   - 沸点
   - 屈服强度
   - 抗拉强度
   - 延展率
   - 热膨胀系数
   - 热值(J/kg)
   - 杨氏模量(GPa)
   - 硬度
   - 疲劳强度
   - 冲击韧性(J/cm²)

原始数据集的第2-105列为材料种类的独热编码。

## 系统特点

1. **部分特征预测**: 系统支持用户仅提供部分已知特征进行预测，未知特征将自动使用训练数据均值填充
2. **多模态融合**: 同时处理分类特征（颜色）和数值特征（物理特性）
3. **高精度识别**: 在测试集上达到较高的识别准确率
4. **可视化结果**: 提供预测结果的可视化展示，包括前k个最可能的材料及其概率
5. **完整的数据处理流程**: 从数据预处理到模型训练、评估和预测的完整流程
6. **RESTful API接口**: 提供API接口，方便集成到其他系统

## 运行环境

- Python 3.6+
- PyTorch 1.7+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Flask (API服务)

## 操作步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

运行以下命令执行完整的训练流程（数据预处理、模型训练、评估）:

```bash
python run_pipeline.py
```

该脚本将:

- 提取材料名称并保存
- 预处理训练和验证数据
- 训练神经网络模型
- 评估模型性能
- 生成学习曲线和混淆矩阵可视化
- 保存训练好的模型和预处理器

### 3. 使用模型进行预测

运行以下命令使用训练好的模型进行预测:

```bash
python predict.py
```

预测时：

- 可以选择手动输入材料特征（支持部分特征）
- 可以使用示例数据进行预测
- 系统会显示前5个最可能的材料类型及其概率
- 结果将以可视化方式呈现

### 4. 启动API服务

运行以下命令启动API服务:

```bash
# 开发模式启动
python api_service.py --debug

# 生产模式启动
python api_service.py --host localhost --port 5000 --workers 4
```

### 5. 调用API示例

参考`api_client.py`中的示例代码:

```bash
python api_client.py --url http://localhost:5000
```

## API接口说明

### 1. 健康检查

- **URL**: `http://localhost:5000/api/health`
- **方法**: GET
- **描述**: 检查API服务是否正常运行
- **返回示例**:

  ```json
  {
    "status": "ok",
    "message": "金属材料多模态识别系统API正常运行中"
  }
  ```

### 2. 获取已知颜色列表

- **URL**: `http://localhost:5000/api/known_colors`
- **方法**: GET
- **描述**: 获取训练数据中已知的颜色列表
- **返回示例**:

  ```json
  {
    "colors": ["银白色", "金黄色", "紫红色", "黑色", "灰色", "褐色"]
  }
  ```

### 3. 预测材料类型

- **URL**: `/api/predict`
- **方法**: POST
- **描述**: 根据提供的特征预测材料类型
- **请求参数**:

  ```json
  {
    "color": "紫红色",         // 必填，材料颜色
    "density": 8.96,          // 可选，密度(g/cm³)
    "resistivity": 1.678,     // 可选，电阻率
    "specific_heat": 0.39,    // 可选，比热容
    "melting_point": 1083,    // 可选，熔点
    "boiling_point": 2567,    // 可选，沸点
    "yield_strength": 220,    // 可选，屈服强度
    "tensile_strength": 240,  // 可选，抗拉强度
    "elongation": 35,         // 可选，延展率
    "thermal_expansion": 1.485e-05,  // 可选，热膨胀系数
    "heat_value": 24160,      // 可选，热值(J/kg)
    "youngs_modulus": 130,    // 可选，杨氏模量(GPa)
    "hardness": 40,           // 可选，硬度
    "fatigue_strength": 139.3,  // 可选，疲劳强度
    "impact_toughness": 42.5  // 可选，冲击韧性(J/cm²)
  }
  ```

- **返回示例**:

  ```json
  {
    "success": true,
    "results": [
      {
        "material": "铜",
        "probability": 0.8923,
        "index": 1
      },
      {
        "material": "紫铜",
        "probability": 0.0734,
        "index": 43
      },
      {
        "material": "铜锌合金",
        "probability": 0.0256,
        "index": 87
      }
    ],
    "result_image": "base64编码的图像..."
  }
  ```

## 技术实现细节

### 数据预处理

1. **颜色特征处理**:
   - 使用标签编码器(LabelEncoder)将文本颜色转换为数值

2. **数值特征处理**:
   - 使用SimpleImputer填充缺失值
   - 使用StandardScaler进行标准化

3. **预测时的缺失值处理**:
   - 仅要求颜色特征必须提供
   - 缺失的数值特征使用训练集均值自动填充

### 模型架构

1. **神经网络结构**:
   - 多层感知器(MLP)
   - 输入层: 15个神经元(1个颜色特征 + 14个数值特征)
   - 隐藏层: [256, 128, 64]个神经元
   - 输出层: 材料种类数量的神经元
   - 激活函数: ReLU
   - 正则化: Dropout(0.3)和BatchNorm

2. **训练参数**:
   - 优化器: Adam
   - 学习率: 0.001
   - 学习率调度: ReduceLROnPlateau
   - 早停机制: patience=15
   - 批量大小: 32

### 评估指标

- 准确率(Accuracy)
- 混淆矩阵(Confusion Matrix)
- 分类报告(Classification Report): 精确率、召回率、F1分数

## 使用示例

1. **完整特征预测**:
   - 提供所有15个特征进行预测
   - 系统返回最可能的材料类型及概率

2. **部分特征预测**:
   - 仅提供颜色和少量数值特征(如密度)
   - 系统自动处理缺失值并返回预测结果

3. **API调用预测**:
   - 通过RESTful API接口调用预测功能
   - 支持与其他系统集成

## 改进方向

1. **模型增强**:
   - 尝试更复杂的网络结构(如残差网络)
   - 探索注意力机制以更好地融合多模态特征

2. **数据增强**:
   - 扩充训练数据集
   - 增加数据增强技术提高模型泛化能力

3. **特征工程**:
   - 尝试不同的特征组合和交互项
   - 探索不同的特征重要性分析方法

4. **集成学习**:
   - 实现模型集成以提高预测精度
   - 结合不同类型的模型(如决策树与神经网络)

5. **API增强**:
   - 添加批量预测功能
   - 增加用户认证和访问控制
   - 实现异步处理大规模请求

## 开发团队

中国矿业大学 计算机科学与技术学院 王欣烁

## 许可证

项目遵循 MIT 许可证
