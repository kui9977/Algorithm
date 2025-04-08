import pandas as pd
import numpy as np
import os
import argparse
import sys
import chardet
import re

def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']

def load_material_data(file_path):
    """
    加载材料数据，并进行必要的预处理
    
    参数:
        file_path: 数据文件路径(.csv或.xlsx)
        
    返回:
        处理后的DataFrame
    """
    print(f"加载材料数据文件: {file_path}")
    
    try:
        # 根据文件类型选择加载方法
        if file_path.lower().endswith('.csv'):
            # 尝试不同的编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'iso-8859-1', 'latin1']
            
            # 先尝试自动检测编码
            detected_encoding = detect_encoding(file_path)
            if detected_encoding and detected_encoding not in encodings:
                encodings.insert(0, detected_encoding)
                
            # 尝试所有可能的编码
            for encoding in encodings:
                try:
                    print(f"尝试使用 {encoding} 编码读取...")
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"成功使用 {encoding} 编码读取文件")
                    break
                except UnicodeDecodeError:
                    print(f"使用 {encoding} 编码失败")
                    continue
                except Exception as e:
                    print(f"读取CSV文件时出错: {e}")
                    continue
            else:
                raise ValueError("无法读取CSV文件，请检查文件编码")
                
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
            print(f"成功读取Excel文件")
        else:
            raise ValueError(f"不支持的文件格式: {file_path}，请使用.csv或.xlsx/.xls格式")
            
        # 文件基本信息
        print(f"数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()[:10]}...")  # 只显示前10个列名
        
        # 处理独热编码列和标签
        onehot_cols = [str(i) for i in range(2, 106) if str(i) in df.columns]
        if onehot_cols:
            print(f"发现 {len(onehot_cols)} 个独热编码列")
            
            # 如果没有label列，从独热编码创建
            if 'label' not in df.columns:
                print("从独热编码创建label列...")
                df['label'] = df[onehot_cols].idxmax(axis=1).astype(int) - 2
                print(f"创建的标签值范围: {df['label'].min()} - {df['label'].max()}")
        else:
            print("警告: 未发现独热编码列")
            
        # 检查和处理必要的特征列
        required_features = ['颜色', '密度（g/cm3）', '温度', '电阻率', '电阻温度系数', 
                           '比热容', '熔点', '沸点', '屈服强度', '抗拉强度', '延展率', 
                           '热膨胀系数', '热值（J/kg）', '杨氏模量GPa', '硬度', 
                           '疲劳强度', '冲击韧性J/cm2']
                           
        # 检查是否存在这些列，并尝试重命名为标准名称
        # 处理列名(密度可能是'密度'或'密度（g/cm3）'等)
        if '密度' in df.columns and '密度（g/cm3）' not in df.columns:
            df.rename(columns={'密度': '密度（g/cm3）'}, inplace=True)
            
        if '杨氏模量' in df.columns and '杨氏模量GPa' not in df.columns:
            df.rename(columns={'杨氏模量': '杨氏模量GPa'}, inplace=True)
            
        if '热值' in df.columns and '热值（J/kg）' not in df.columns:
            df.rename(columns={'热值': '热值（J/kg）'}, inplace=True)
            
        if '冲击韧性' in df.columns and '冲击韧性J/cm2' not in df.columns:
            df.rename(columns={'冲击韧性': '冲击韧性J/cm2'}, inplace=True)
        
        # 检查颜色列是否存在
        if '颜色' not in df.columns:
            print("警告: 找不到'颜色'列，这是必需的特征")
            
        # 统一使用短列名
        rename_dict = {
            '密度（g/cm3）': '密度',
            '热值（J/kg）': '热值',
            '杨氏模量GPa': '杨氏模量',
            '冲击韧性J/cm2': '冲击韧性'
        }
        df.rename(columns=rename_dict, inplace=True)
        
        # 预览处理后的数据
        print("\n处理后的数据预览:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def prepare_training_data(file_path, output_path=None):
    """
    准备训练数据，包括数据清洗、特征处理等
    
    参数:
        file_path: 原始数据文件路径
        output_path: 处理后数据保存路径(可选)
        
    返回:
        处理后的DataFrame
    """
    # 加载数据
    df = load_material_data(file_path)
    if df is None:
        return None
        
    # 数据清洗
    # 1. 填充缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(-1)
    
    # 2. 移除重复行
    before_drop = len(df)
    df.drop_duplicates(inplace=True)
    after_drop = len(df)
    if before_drop > after_drop:
        print(f"移除了 {before_drop - after_drop} 行重复数据")
    
    # 3. 检查并处理异常值
    for col in numeric_cols:
        if col in df.columns and col != 'label':
            # 计算Z分数
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            # 找出Z分数大于3的值
            outliers = (z_scores > 3)
            if outliers.sum() > 0:
                print(f"列 '{col}' 中发现 {outliers.sum()} 个异常值，替换为-1")
                df.loc[outliers, col] = -1
    
    # 如果需要，保存处理后的数据
    if output_path:
        # 确定保存格式
        if output_path.lower().endswith('.csv'):
            df.to_csv(output_path, index=False, encoding='utf-8')
        elif output_path.lower().endswith(('.xlsx', '.xls')):
            df.to_excel(output_path, index=False)
        else:
            # 默认使用CSV格式
            output_path = output_path + '.csv' if '.' not in output_path else output_path
            df.to_csv(output_path, index=False, encoding='utf-8')
            
        print(f"处理后的数据已保存至: {output_path}")
    
    return df

def preprocess_training_data(file_path, output_path=None):
    """
    预处理training_data.csv文件，处理独热编码和特征列
    
    参数:
        file_path: 输入文件路径
        output_path: 处理后的输出文件路径(可选)
        
    返回:
        处理后的DataFrame
    """
    print(f"正在预处理文件: {file_path}")
    
    # 检测文件编码
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))
        encoding = result['encoding']
        print(f"检测到文件编码: {encoding}, 置信度: {result['confidence']:.2f}")
    except Exception as e:
        print(f"检测文件编码时出错: {e}")
        encoding = 'utf-8'  # 默认使用utf-8
    
    # 读取文件
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"成功读取文件，形状: {df.shape}")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None
    
    # 处理列名，移除可能导致问题的空格
    df.columns = [col.strip() for col in df.columns]
    
    # 1. 处理独热编码
    onehot_cols = [str(i) for i in range(2, 106) if str(i) in df.columns]
    if onehot_cols:
        print(f"发现{len(onehot_cols)}个独热编码列")
        # 填充可能的NaN值为0
        df[onehot_cols] = df[onehot_cols].fillna(0)
        
        # 创建标签列
        try:
            # 找出每行中值为1的列的索引
            df['label'] = df[onehot_cols].idxmax(axis=1).astype(int) - 2
            print(f"成功创建标签列，唯一值: {sorted(df['label'].unique())}")
            
            # 检查是否有所有值都为0的行
            no_class_rows = (df[onehot_cols].sum(axis=1) == 0)
            if no_class_rows.any():
                print(f"警告: 发现{no_class_rows.sum()}行没有类别信息（所有独热编码值为0）")
                df.loc[no_class_rows, 'label'] = -1
        except Exception as e:
            print(f"创建标签列时出错: {e}")
            # 创建一个空的标签列
            df['label'] = -1
    else:
        print("未找到独热编码列")
        df['label'] = -1
    
    # 2. 处理特征列
    # 标准化特征列名称
    rename_dict = {}
    for col in df.columns:
        # 去除列名中的空格
        new_col = col.strip()
        if col != new_col:
            rename_dict[col] = new_col
    
    if rename_dict:
        df = df.rename(columns=rename_dict)
        print(f"已重命名{len(rename_dict)}个列名")
    
    # 3. 处理数值特征
    numerical_cols = [col for col in df.columns if col not in ['名称', '颜色', 'label'] 
                      and col not in onehot_cols]
    
    print(f"正在处理{len(numerical_cols)}个数值特征列")
    
    # 处理包含特殊字符的数值
    for col in numerical_cols:
        if df[col].dtype == 'object':
            print(f"转换列 '{col}' 为数值型")
            # 使用正则表达式提取数值部分
            df[col] = df[col].apply(lambda x: extract_number(x) if isinstance(x, str) else x)
            # 转换为数值类型
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 填充缺失值
    df[numerical_cols] = df[numerical_cols].fillna(-1)
    
    # 4. 保存处理后的数据
    if output_path:
        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"已将处理后的数据保存到: {output_path}")
        except Exception as e:
            print(f"保存数据时出错: {e}")
    
    return df

def extract_number(text):
    """从文本中提取数值"""
    if not text or pd.isna(text):
        return np.nan
    
    # 尝试提取第一个数值
    numbers = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', str(text))
    if numbers:
        return float(numbers[0])
    
    # 如果找不到数字，返回NaN
    return np.nan

def main():
    """命令行入口函数"""
    if len(sys.argv) < 2:
        print("使用方法: python prepare_data.py <input_file> [<output_file>]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # 如果未指定输出文件，创建一个默认名称
    if not output_file:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_processed.csv"
    
    # 处理数据
    preprocess_training_data(input_file, output_file)

if __name__ == "__main__":
    main()
