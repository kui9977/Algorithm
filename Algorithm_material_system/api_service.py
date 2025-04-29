#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
from waitress import serve
import logging
import sys

# 确保可以导入api模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 导入api模块前确保它存在
api_module_path = os.path.join(current_dir, 'api.py')
if not os.path.exists(api_module_path):
    print(f"错误: 无法找到api模块文件 '{api_module_path}'")
    print("请先创建api.py文件，定义API接口")
    sys.exit(1)

try:
    from api import app
except ImportError as e:
    print(f"导入api模块失败: {e}")
    print("请确保api.py文件中定义了app变量")
    sys.exit(1)

def setup_logging(debug=False):
    """配置日志记录"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='金属材料多模态识别系统API服务')
    parser.add_argument('--debug', action='store_true', 
                        help='启用调试模式')
    parser.add_argument('--host', type=str, default='localhost', 
                        help='服务监听地址 (默认: localhost)')
    parser.add_argument('--port', type=int, default=5000, 
                        help='服务监听端口 (默认: 5000)')
    parser.add_argument('--workers', type=int, default=4, 
                        help='工作进程数 (仅生产模式有效, 默认: 4)')
    return parser.parse_args()

def main():
    """主函数，启动API服务"""
    args = parse_args()
    logger = setup_logging(args.debug)
    
    if args.debug:
        logger.info(f"以调试模式启动API服务，地址: {args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=True)
    else:
        logger.info(f"以生产模式启动API服务，地址: {args.host}:{args.port}，工作进程数: {args.workers}")
        print(f"API服务运行在 http://{args.host}:{args.port}")
        serve(app, host=args.host, port=args.port, threads=args.workers)

if __name__ == "__main__":
    main()
