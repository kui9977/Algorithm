import pandas as pd
import pickle
import os

def extract_and_save_material_names(train_path='./complete_materials_train.csv', 
                                   val_path='./complete_materials_val.csv'):
    """提取并保存材料名称列表"""
    # 尝试多种编码方式读取数据
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'cp936', 'latin-1', 'iso-8859-1']
    
    train_df = None
    val_df = None
    
    # 尝试自动检测编码
    try:
        import chardet
        with open(train_path, 'rb') as f:
            raw_data = f.read(10000)  # 读取前10000字节
            result = chardet.detect(raw_data)
            detected_encoding = result['encoding']
            confidence = result['confidence']
            print(f"检测到文件编码可能为 {detected_encoding}，置信度: {confidence}")
            
            # 如果检测到编码且置信度高，优先使用检测到的编码
            if detected_encoding and confidence > 0.7:
                try:
                    train_df = pd.read_csv(train_path, encoding=detected_encoding)
                    val_df = pd.read_csv(val_path, encoding=detected_encoding)
                    print(f"成功使用检测到的编码 {detected_encoding} 读取数据!")
                except Exception as e:
                    print(f"使用检测到的编码 {detected_encoding} 读取失败: {e}")
    except ImportError:
        print("未安装chardet库，无法自动检测编码")
    
    # 如果自动检测失败，尝试所有指定的编码
    if train_df is None or val_df is None:
        for encoding in encodings:
            try:
                print(f"尝试使用 {encoding} 编码读取数据...")
                train_df = pd.read_csv(train_path, encoding=encoding)
                val_df = pd.read_csv(val_path, encoding=encoding)
                print(f"成功使用 {encoding} 编码读取数据!")
                break
            except Exception as e:
                print(f"使用 {encoding} 编码读取失败: {e}")
                continue
    
    if train_df is None or val_df is None:
        raise ValueError("无法读取数据文件，请检查文件格式和编码。")
    
    # 提取材料名称
    material_names = []
    
    # 从训练集中提取材料名称
    for i in range(train_df.shape[0]):
        name = train_df.iloc[i, 0]
        material_names.append(name)
    
    # 从验证集中提取材料名称
    for i in range(val_df.shape[0]):
        name = val_df.iloc[i, 0]
        material_names.append(name)
    
    # 删除重复的名称
    material_names = sorted(list(set(material_names)))
    
    # 保存材料名称
    os.makedirs('models', exist_ok=True)
    with open('models/material_names.pkl', 'wb') as f:
        pickle.dump(material_names, f)
    
    print(f"共提取并保存了 {len(material_names)} 个材料名称")
    
    # 显示部分材料名称
    print("材料名称示例:")
    for i, name in enumerate(material_names[:10]):
        print(f"{i+1}. {name}")
    print("...")
    
    return material_names

if __name__ == "__main__":
    extract_and_save_material_names()
