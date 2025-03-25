import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch
import logging
from datetime import datetime

from predict import MaterialPredictor
from train import train_model, evaluate_model
from models.model import MaterialAnalysisModel
from data.preprocessor import MaterialDataPreprocessor, create_dataloaders

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("material_api")

app = Flask(__name__)

# 全局配置
MODEL_DIR = os.environ.get('MODEL_DIR', 'models')
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# 确保目录存在
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 配置Flask应用
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 全局变量
predictor = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_data_from_file(file_path):
    """
    从文件加载数据，支持CSV和Excel格式

    参数:
        file_path: 数据文件路径

    返回:
        pandas DataFrame: 加载的数据

    异常:
        ValueError: 当文件格式不支持时
        Exception: 当文件加载出错时
    """
    try:
        # 根据文件扩展名选择适当的加载方法
        if file_path.endswith('.csv'):
            # 尝试不同的编码格式
            encodings = ['utf-8', 'gbk', 'gb2312', 'iso-8859-1', 'latin1']
            for encoding in encodings:
                try:
                    return pd.read_csv(file_path, encoding=encoding)
                except UnicodeDecodeError as e:
                    logger.warning(
                        f"尝试使用{encoding}编码读取时发生UnicodeDecodeError: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"尝试使用{encoding}编码读取时出错: {e}")

            # 如果所有编码都失败，尝试自动检测编码
            import chardet
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read())
            detected_encoding = result.get('encoding')
            if detected_encoding:
                logger.info(f"检测到文件编码: {detected_encoding}")
                return pd.read_csv(file_path, encoding=detected_encoding)
            else:
                raise ValueError("无法检测文件编码，请检查文件内容是否正确")

        elif file_path.endswith(('.xlsx', '.xls')):
            # Excel文件不需要指定编码
            return pd.read_excel(file_path, engine='openpyxl')
        else:
            raise ValueError(f"不支持的文件格式: {file_path}，请使用.csv或.xlsx/.xls格式")
    except Exception as e:
        logger.error(f"加载数据文件时出错: {e}")
        raise


def init_predictor():
    """初始化预测器"""
    global predictor

    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    metadata_path = os.path.join(MODEL_DIR, "metadata.json")

    # 检查必要文件是否存在
    if not all(os.path.exists(p) for p in [model_path, preprocessor_path, metadata_path]):
        logger.warning("预测器初始化失败：模型文件不存在")
        return False

    try:
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        predictor = MaterialPredictor(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            metadata_path=metadata_path,
            device=device
        )
        logger.info(f"预测器已初始化，使用设备: {device}")
        return True
    except Exception as e:
        logger.error(f"初始化预测器时出错: {e}")
        return False


@app.route('/api/predict', methods=['POST'])
def predict_api():
    """批量预测API接口"""
    # 检查预测器是否已加载
    global predictor
    if predictor is None:
        success = init_predictor()
        if not success:
            return jsonify({
                'status': 'error',
                'message': '预测模型未初始化'
            }), 500

    # 检查是否是JSON请求还是文件上传
    if request.is_json:
        # 处理JSON请求
        try:
            data = request.get_json()
            # 转换为DataFrame
            df = pd.DataFrame(data['samples'])
            logger.info(f"通过JSON接收到{len(df)}条预测数据")
        except Exception as e:
            logger.error(f"解析JSON数据时出错: {e}")
            return jsonify({
                'status': 'error',
                'message': f'无效的JSON数据: {str(e)}'
            }), 400
    else:
        # 处理文件上传
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': '没有上传文件'
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': '未选择文件'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'status': 'error',
                'message': '不支持的文件格式，请上传CSV或Excel文件'
            }), 400

        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            logger.info(f"文件已保存: {file_path}")

            # 加载数据
            df = load_data_from_file(file_path)
            logger.info(f"从文件加载了{len(df)}条数据")
        except Exception as e:
            logger.error(f"处理上传文件时出错: {e}")
            return jsonify({
                'status': 'error',
                'message': f'处理文件出错: {str(e)}'
            }), 500

    # 执行预测
    try:
        pred_classes, pred_probs, class_names = predictor.predict_batch(df)

        # 准备结果
        results = []
        for i in range(len(df)):
            results.append({
                'id': i,
                'predicted_class': int(pred_classes[i]),
                'class_name': class_names[i],
                'confidence': float(pred_probs[i])
            })

        return jsonify({
            'status': 'success',
            'predictions': results,
            'count': len(results)
        })

    except Exception as e:
        logger.error(f"预测过程出错: {e}")
        return jsonify({
            'status': 'error',
            'message': f'预测过程出错: {str(e)}'
        }), 500


@app.route('/api/train', methods=['POST'])
def train_api():
    """训练API接口"""
    # 检查是否是文件上传
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': '没有上传训练数据文件'
        }), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': '未选择文件'
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': '不支持的文件格式，请上传CSV或Excel文件'
        }), 400

    # 获取训练参数
    batch_size = int(request.form.get('batch_size', 32))
    epochs = int(request.form.get('epochs', 30))
    lr = float(request.form.get('lr', 0.001))
    patience = int(request.form.get('patience', 5))
    train_ratio = float(request.form.get('train_ratio', 0.7))
    val_ratio = float(request.form.get('val_ratio', 0.15))

    try:
        # 保存上传的文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logger.info(f"训练文件已保存: {file_path}")

        # 加载数据
        df = load_data_from_file(file_path)
        logger.info(f"从文件加载了{len(df)}条训练数据")

        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")

        # 执行训练流程
        # 创建预处理器
        preprocessor = MaterialDataPreprocessor()

        # 创建数据加载器
        train_loader, val_loader, test_loader = create_dataloaders(
            df, preprocessor, batch_size=batch_size,
            train_ratio=train_ratio, val_ratio=val_ratio
        )

        # 获取特征数量和类别数量
        num_colors = preprocessor.get_vocab_size()
        num_classes = len(df['label'].unique())
        num_numerical_features = len(
            [col for col in df.columns if col not in ['颜色', 'label']])

        logger.info(
            f"颜色数量: {num_colors}, 类别数量: {num_classes}, 数值特征数量: {num_numerical_features}")

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
            model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )

        # 训练模型
        logger.info("开始训练模型...")
        model = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs=epochs, device=device, patience=patience
        )

        # 评估模型
        logger.info("在测试集上评估模型...")
        accuracy, f1, _ = evaluate_model(
            model, test_loader, criterion, device=device
        )

        # 保存训练产物
        from predict import save_training_artifacts

        # 创建类别映射
        class_indices = df['label'].unique()
        class_mapping = {int(idx): f"类别{idx}" for idx in class_indices}

        # 数值特征列
        numerical_columns = [
            col for col in df.columns if col not in ['颜色', 'label']]

        # 保存训练产物到指定MODEL_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_DIR, f"best_model_{timestamp}.pth")

        # 保存模型权重
        torch.save(model.state_dict(), model_path)

        # 保存为标准名称以供API使用
        standard_model_path = os.path.join(MODEL_DIR, "best_model.pth")
        torch.save(model.state_dict(), standard_model_path)

        # 保存其他训练产物
        save_training_artifacts(
            model, preprocessor, class_mapping, numerical_columns,
            model_dir=MODEL_DIR
        )

        # 清除预测器以便重新加载
        global predictor
        predictor = None

        return jsonify({
            'status': 'success',
            'message': '模型训练完成',
            'metrics': {
                'accuracy': float(accuracy),
                'f1_score': float(f1)
            },
            'model_info': {
                'num_classes': num_classes,
                'num_colors': num_colors,
                'num_features': num_numerical_features
            }
        })

    except Exception as e:
        logger.error(f"训练过程出错: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'训练过程出错: {str(e)}'
        }), 500


@app.route('/api/model_status', methods=['GET'])
def model_status():
    """获取模型状态"""
    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")
    metadata_path = os.path.join(MODEL_DIR, "metadata.json")

    files_exist = all(os.path.exists(p)
                      for p in [model_path, preprocessor_path, metadata_path])

    # 如果有元数据文件，读取更多信息
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except:
            metadata = {}

    return jsonify({
        'status': 'ready' if files_exist else 'not_ready',
        'files': {
            'model': os.path.exists(model_path),
            'preprocessor': os.path.exists(preprocessor_path),
            'metadata': os.path.exists(metadata_path)
        },
        'model_info': metadata.get('model_info', {}),
        'class_mapping': metadata.get('class_mapping', {})
    })


@app.route('/api/predict_single', methods=['POST'])
def predict_single():
    """单样本预测API接口"""
    # 检查预测器是否已加载
    global predictor
    if predictor is None:
        success = init_predictor()
        if not success:
            return jsonify({
                'status': 'error',
                'message': '预测模型未初始化'
            }), 500

    # 获取单样本数据
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            return jsonify({
                'status': 'error',
                'message': '无效的输入数据格式'
            }), 400

        # 转换为DataFrame
        sample_df = pd.DataFrame([data])

        # 执行预测
        pred_class, pred_prob, class_name = predictor.predict(sample_df)

        # 尝试获取解释信息
        try:
            explanation = predictor.explain_prediction(sample_df)
        except:
            explanation = {
                'predicted_class': int(pred_class),
                'class_name': class_name,
                'confidence': float(pred_prob)
            }

        return jsonify({
            'status': 'success',
            'prediction': {
                'class': int(pred_class),
                'class_name': class_name,
                'confidence': float(pred_prob)
            },
            'explanation': explanation
        })

    except Exception as e:
        logger.error(f"单样本预测时出错: {e}")
        return jsonify({
            'status': 'error',
            'message': f'预测过程出错: {str(e)}'
        }), 500


if __name__ == '__main__':
    # 尝试初始化预测器
    init_predictor()

    # 启动Flask服务
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'

    app.run(host='0.0.0.0', port=port, debug=debug)
