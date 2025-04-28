"""
金属材料多模态识别系统 API 服务
负责启动和管理API服务
"""

import os
import sys
import time
import argparse
import logging
from waitress import serve
from api import app

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MetalAPI")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="金属材料多模态识别系统 API 服务")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务主机地址')
    parser.add_argument('--port', type=int, default=5000, help='服务端口')
    parser.add_argument('--workers', type=int, default=4, help='工作进程数')
    parser.add_argument('--debug', action='store_true', help='是否开启调试模式')
    return parser.parse_args()

def start_development_server(host, port):
    """启动开发服务器"""
    logger.info(f"启动开发服务器: {host}:{port}")
    app.run(host=host, port=port, debug=True)

def start_production_server(host, port, workers):
    """启动生产服务器"""
    logger.info(f"启动生产服务器: {host}:{port} (工作进程: {workers})")
    serve(app, host=host, port=port, threads=workers)

def main():
    """主函数"""
    args = parse_arguments()
    
    # 检查系统环境
    logger.info(f"Python 版本: {sys.version}")
    logger.info(f"当前工作目录: {os.getcwd()}")
    
    # 检查模型文件是否存在
    required_files = [
        'models/metal_classifier.pth',
        'models/preprocessor.pkl',
        'models/material_names.pkl'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        logger.error(f"缺少必要的模型文件: {missing_files}")
        logger.error("请先运行 run_pipeline.py 完成模型训练!")
        return
    
    logger.info("模型文件检查完成，所有必要文件都存在")
    
    try:
        # 根据调试标志选择服务器类型
        if args.debug:
            start_development_server(args.host, args.port)
        else:
            start_production_server(args.host, args.port, args.workers)
    except KeyboardInterrupt:
        logger.info("接收到终止信号，服务停止")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
