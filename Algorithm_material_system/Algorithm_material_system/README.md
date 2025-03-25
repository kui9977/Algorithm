# 基于多模态学习的复合金属材料智能分析识别系统

本项目实现了一个基于多模态学习的复合金属材料分析识别系统，能够通过整合材料的颜色信息（文本）和物理属性（数值）来识别材料类型并分析其性质。系统支持通过API调用，可作为网站的后端服务。

## 系统功能

1. 材料分类：根据材料的颜色和物理属性识别材料的类型
2. 特性分析：分析材料的各类性质及适用场景
3. 批量预测：支持批量材料数据的处理和识别
4. API服务：提供RESTful API接口，支持模型训练、预测和状态查询

## 模型架构

系统采用多模态融合架构，包括：

1. **颜色特征处理分支**：将颜色文本通过嵌入层转换为稠密向量表示
2. **数值特征处理分支**：使用多层感知器处理材料的物理属性数据
3. **Transformer模型**：通过多头自注意力机制实现模态间的交互和信息融合
4. **分类层**：基于融合特征进行材料类型识别

## 项目结构

```python
.
├── models/                    # 模型定义
│   └── model.py              # 多模态融合模型架构
├── data/                     # 数据处理
│   └── preprocessor.py       # 数据预处理工具
├── train.py                  # 模型训练脚本
├── predict.py                # 模型预测和部署
├── run_pipeline.py           # 完整处理流程
├── api.py                    # API接口实现
├── server.py                 # 服务器入口
└── README.md                 # 项目说明文档
```

## 安装依赖

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn tqdm flask werkzeug chardet
```

> **注意**: 安装`chardet`库以支持自动检测文件编码，解决编码问题。

## 文件编码说明

系统支持多种文件编码格式：

1. 优先尝试以下编码读取CSV文件：
   - UTF-8
   - GBK（中文Windows常用编码）
   - GB2312（简体中文编码）
   - ISO-8859-1（拉丁字符编码）
   - Latin1

2. 如果上述编码均失败，系统将使用`chardet`库自动检测文件编码。

3. Excel文件(.xlsx, .xls)不需要指定编码，可直接读取。

4. 如遇到编码错误，可尝试以下解决方法：
   - 使用文本编辑器（如Notepad++）将文件转换为UTF-8编码
   - 安装chardet库：`pip install chardet`
   - 手动指定正确的文件编码

## 使用方法

### 1. 命令行方式

#### 训练模型

```bash
python run_pipeline.py --mode train --data your_data.csv --model_dir models
```

#### 预测材料类型

```bash
python run_pipeline.py --mode predict --data your_data.csv --predict_input new_samples.csv --predict_output predictions.csv
```

#### 评估模型性能

```bash
python run_pipeline.py --mode evaluate --data test_data.csv --model_dir models
```

### 2. API服务方式

#### 启动API服务器

```bash
python server.py --port 5000 --model_dir models --upload_dir uploads
```

#### API接口说明

1. **模型状态查询**
   - 端点：`GET /api/model_status`
   - 功能：查询模型是否已加载及相关信息
   - 响应示例：

     ```json
     {
       "status": "ready",
       "files": {
         "model": true,
         "preprocessor": true,
         "metadata": true
       },
       "model_info": {
         "num_classes": 5,
         "num_features": 15,
         "num_colors": 10
       }
     }
     ```

2. **单样本预测**
   - 端点：`POST /api/predict_single`
   - 功能：预测单个材料样本的类别
   - 请求体：材料属性的JSON对象

     ```json
     {
       "颜色": "银色",
       "密度": 7.85,
       "电阻率": 9.71e-8,
       "电阻温度系数": 0.0065,
       "比热容": 450,
       "熔点": 1538,
       "沸点": 2862,
       "屈服强度": 250,
       "抗拉强度": 420,
       "延展率": 22,
       "热膨胀系数": 1.2e-5,
       "热值": 0,
       "杨氏模量": 210,
       "硬度": 150,
       "疲劳强度": 170,
       "冲击韧性": 30
     }
     ```

   - 响应示例：

     ```json
     {
       "status": "success",
       "prediction": {
         "class": 2,
         "class_name": "碳钢",
         "confidence": 0.89
       },
       "explanation": {
         "predicted_class": 2,
         "class_name": "碳钢",
         "confidence": 0.89,
         "typical_properties": {},
         "possible_applications": []
       }
     }
     ```

3. **批量预测 (JSON格式)**
   - 端点：`POST /api/predict`
   - 功能：预测多个材料样本的类别
   - 请求体：包含多个材料样本的JSON对象数组

     ```json
     {
       "samples": [
         {
           "颜色": "银色",
           "密度": 7.85,
           "电阻率": 9.71e-8,
           ...
         },
         {
           "颜色": "金色",
           "密度": 8.96,
           "电阻率": 1.68e-8,
           ...
         }
       ]
     }
     ```

   - 响应示例：

     ```json
     {
       "status": "success",
       "predictions": [
         {
           "id": 0,
           "predicted_class": 2,
           "class_name": "碳钢",
           "confidence": 0.89
         },
         {
           "id": 1,
           "predicted_class": 3,
           "class_name": "铜合金",
           "confidence": 0.95
         }
       ],
       "count": 2
     }
     ```

4. **批量预测 (文件上传)**
   - 端点：`POST /api/predict`
   - 功能：通过上传CSV或Excel文件进行批量预测
   - 请求：使用multipart/form-data上传文件，文件字段名为"file"

5. **模型训练**
   - 端点：`POST /api/train`
   - 功能：通过上传数据文件训练新模型
   - 请求：使用multipart/form-data上传文件，文件字段名为"file"
   - 可选表单参数：
     - `batch_size`：批量大小，默认32
     - `epochs`：训练轮数，默认30
     - `lr`：学习率，默认0.001
     - `patience`：早停耐心值，默认5
     - `train_ratio`：训练集比例，默认0.7
     - `val_ratio`：验证集比例，默认0.15

## 数据格式

系统输入的CSV文件应包含以下字段：

1. `颜色`：材料的颜色描述（文本）
2. 数值特征：
   - `密度`
   - `电阻率`
   - `电阻温度系数`
   - `比热容`
   - `熔点`
   - `沸点`
   - `屈服强度`
   - `抗拉强度`
   - `延展率`
   - `热膨胀系数`
   - `热值`
   - `杨氏模量`
   - `硬度`
   - `疲劳强度`
   - `冲击韧性`
3. `label`：材料类别（训练时需要）

## 模型训练参数

- 批量大小: 32
- 学习率: 0.001
- 训练轮数: 30
- 早停耐心值: 5
- 优化器: AdamW
- 学习率调度: CosineAnnealingLR

## 部署建议

### 作为网站后端

1. 使用Nginx等反向代理服务器，将API请求转发到Flask服务器
2. 结合Gunicorn或uWSGI实现生产环境部署
3. 使用Docker容器化部署，简化环境配置

### 示例Nginx配置

```
server {
    listen 80;
    server_name yoursite.com;

    location /api/ {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location / {
        root /path/to/your/frontend;
        index index.html;
    }
}
```

### 示例Docker部署

```bash
# 构建Docker镜像
docker build -t material-analysis-api .

# 运行容器
docker run -d -p 5000:5000 -v /path/to/models:/app/models -v /path/to/uploads:/app/uploads material-analysis-api
```

## 扩展功能

- JSON文件提供类别名称映射
- 支持导出模型解释信息，包括材料典型属性和应用场景
- 支持批量预测并输出为CSV格式
- API服务化，便于集成到现有系统
