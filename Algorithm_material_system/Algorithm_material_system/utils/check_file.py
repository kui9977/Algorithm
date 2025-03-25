import os
import pandas as pd
import chardet
import argparse
import sys


def detect_file_encoding(file_path):
    """检测文件编码并显示信息"""
    print(f"检查文件: {file_path}")
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return

    # 获取文件信息
    file_size = os.path.getsize(file_path) / 1024  # KB
    print(f"文件大小: {file_size:.2f} KB")

    # 检测文件类型
    if file_path.lower().endswith(('.xlsx', '.xls')):
        print(f"检测到Excel文件: {file_path}")
        try:
            # 尝试读取
            if file_path.lower().endswith('.xlsx'):
                df = pd.read_excel(file_path, engine='openpyxl')
                print("成功使用openpyxl引擎读取文件")
            else:
                df = pd.read_excel(file_path, engine='xlrd')
                print("成功使用xlrd引擎读取文件")

            print(f"数据形状: {df.shape}")
            print(f"列名: {df.columns.tolist()}")
            print(f"前5行数据预览:\n{df.head()}")
            return True
        except Exception as e:
            print(f"读取Excel文件失败: {e}")
            return False
    elif file_path.lower().endswith('.csv'):
        print(f"检测到CSV文件: {file_path}")
        # 检测文件编码
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # 读取前10KB进行编码检测
            result = chardet.detect(raw_data)

        encoding = result['encoding']
        confidence = result['confidence']
        print(f"检测到编码: {encoding}，置信度: {confidence:.2f}")

        # 尝试不同编码读取
        encodings = ['utf-8', 'gbk', 'gb2312', 'iso-8859-1',
                     'latin1', 'utf-16', 'cp936', 'big5']
        if encoding and encoding not in encodings:
            encodings.insert(0, encoding)

        for enc in encodings:
            try:
                print(f"尝试使用 {enc} 编码读取...")
                df = pd.read_csv(file_path, encoding=enc)
                print(f"成功使用 {enc} 编码读取文件")
                print(f"数据形状: {df.shape}")
                print(f"列名: {df.columns.tolist()}")
                print(f"前5行数据预览:\n{df.head()}")
                return True
            except UnicodeDecodeError:
                print(f"使用 {enc} 编码失败")
                continue
            except Exception as e:
                print(f"读取CSV文件时出错: {e}")
                continue

        print("所有编码尝试都失败，建议将文件转换为Excel格式")
        return False
    else:
        print(f"不支持的文件格式: {file_path}")
        return False


def convert_to_excel(file_path, output_path=None):
    """将CSV文件转换为Excel格式"""
    if not file_path.lower().endswith('.csv'):
        print("只支持将CSV文件转换为Excel格式")
        return False

    if output_path is None:
        output_path = os.path.splitext(file_path)[0] + '.xlsx'

    # 尝试检测并读取CSV
    encodings = ['utf-8', 'gbk', 'gb2312', 'iso-8859-1',
                 'latin1', 'utf-16', 'cp936', 'big5']

    # 先用chardet检测
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)

    encoding = result['encoding']
    if encoding and encoding not in encodings:
        encodings.insert(0, encoding)

    # 尝试读取
    for enc in encodings:
        try:
            print(f"尝试使用 {enc} 编码读取...")
            df = pd.read_csv(file_path, encoding=enc)
            print(f"成功使用 {enc} 编码读取文件")

            # 保存为Excel
            df.to_excel(output_path, index=False, engine='openpyxl')
            print(f"已成功将文件转换为Excel格式: {output_path}")
            return True
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"处理文件时出错: {e}")
            continue

    print("无法读取CSV文件，转换失败")
    return False


def main():
    parser = argparse.ArgumentParser(description="文件编码检测和转换工具")
    parser.add_argument("--file", type=str, required=True, help="要检查的文件路径")
    parser.add_argument("--convert", action="store_true",
                        help="是否将CSV转换为Excel")
    parser.add_argument("--output", type=str, help="输出Excel文件路径(可选)")

    args = parser.parse_args()

    # 检查文件编码
    success = detect_file_encoding(args.file)

    # 转换为Excel(如果需要)
    if args.convert and args.file.lower().endswith('.csv'):
        print("\n开始转换文件为Excel格式...")
        convert_to_excel(args.file, args.output)


if __name__ == "__main__":
    main()
