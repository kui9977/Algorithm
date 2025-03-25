import os
import argparse
import logging
from api import app
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("server")


def check_dependencies():
    """检查必要的依赖是否安装"""
    try:
        import chardet
        logger.info("chardet库已安装，可用于自动检测文件编码")
    except ImportError:
        logger.warning("未安装chardet库，建议使用以下命令安装以支持自动检测文件编码：")
        logger.warning("pip install chardet")
        logger.warning("或者使用正确的编码格式手动指定文件编码")

    # 检查Excel支持
    try:
        import openpyxl
        logger.info("openpyxl库已安装，可用于读取Excel (.xlsx)文件")
    except ImportError:
        logger.warning("未安装openpyxl库，建议使用以下命令安装以支持Excel文件：")
        logger.warning("pip install openpyxl")

    try:
        import xlrd
        logger.info("xlrd库已安装，可用于读取旧版Excel (.xls)文件")
    except ImportError:
        logger.warning("未安装xlrd库，建议使用以下命令安装以支持旧版Excel文件：")
        logger.warning("pip install xlrd")

    # 检查文件系统编码
    import sys
    logger.info(f"文件系统默认编码: {sys.getfilesystemencoding()}")
    logger.info(f"标准输出默认编码: {sys.stdout.encoding}")

    # 显示支持的编码列表
    logger.info(
        "系统将尝试以下编码自动检测CSV文件: utf-8, gbk, gb2312, iso-8859-1, latin1, utf-16, cp936, big5")
    logger.info("如遇编码问题，请尝试使用Excel格式(.xlsx)代替CSV文件")


def parse_args():
    parser = argparse.ArgumentParser(description='材料识别系统服务器')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='服务器监听地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                        help='服务器监听端口 (默认: 5000)')
    parser.add_argument('--debug', action='store_true',
                        help='是否启用调试模式')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='模型目录 (默认: models)')
    parser.add_argument('--upload_dir', type=str, default='uploads',
                        help='上传文件目录 (默认: uploads)')

    return parser.parse_args()


def main():
    """启动服务器"""
    # 检查依赖
    check_dependencies()

    args = parse_args()

    # 设置环境变量
    os.environ['MODEL_DIR'] = args.model_dir
    os.environ['UPLOAD_FOLDER'] = args.upload_dir
    if args.debug:
        os.environ['FLASK_DEBUG'] = '1'

    # 确保目录存在
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.upload_dir, exist_ok=True)

    # 输出信息
    logger.info(f"启动材料识别系统服务器 at {args.host}:{args.port}")
    logger.info(f"模型目录: {args.model_dir}")
    logger.info(f"上传文件目录: {args.upload_dir}")

    # 启动Flask应用
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("服务器被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"服务器出现错误: {e}", exc_info=True)
        sys.exit(1)
