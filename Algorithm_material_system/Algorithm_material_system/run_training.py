import os
import subprocess
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='预处理数据并训练模型')
    parser.add_argument('--data', type=str, required=True, 
                        help='训练数据文件路径（CSV格式）')
    parser.add_argument('--no-preprocess', action='store_true',
                        help='跳过数据预处理步骤')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='模型保存目录')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--epochs', type=int, default=30,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 确保模型目录存在
    os.makedirs(args.model_dir, exist_ok=True)
    
    # 数据预处理
    if not args.no_preprocess:
        print("=== 第1步: 数据预处理 ===")
        processed_data = os.path.join(os.path.dirname(args.data), 
                                     os.path.basename(args.data).replace('.csv', '_processed.csv'))
        
        try:
            # 导入预处理模块
            from models.prepare_data import preprocess_training_data
            
            # 执行预处理
            print(f"开始预处理数据: {args.data}")
            df = preprocess_training_data(args.data, processed_data)
            
            if df is None or len(df) == 0:
                print("预处理失败或数据为空，退出程序")
                return
                
            print(f"数据预处理完成，保存到: {processed_data}")
            
            # 更新数据路径
            data_path = processed_data
        except ImportError:
            print("警告: 无法导入预处理模块，将使用原始数据")
            data_path = args.data
        except Exception as e:
            print(f"预处理时出错: {e}")
            print("将使用原始数据继续")
            data_path = args.data
    else:
        print("跳过数据预处理步骤")
        data_path = args.data
    
    # 训练模型
    print("\n=== 第2步: 训练模型 ===")
    cmd = [
        "python", "run_pipeline.py",
        "--mode", "train",
        "--data", data_path,
        "--model_dir", args.model_dir,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr)
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 执行训练脚本
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                  universal_newlines=True, bufsize=1)
        
        # 实时输出日志
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            
        # 等待进程结束
        process.wait()
        
        if process.returncode != 0:
            print(f"训练失败，退出代码: {process.returncode}")
        else:
            print("训练完成!")
    except Exception as e:
        print(f"启动训练时出错: {e}")
    
if __name__ == "__main__":
    main()
