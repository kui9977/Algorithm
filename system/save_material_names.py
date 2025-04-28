import pandas as pd
import pickle
import os
import numpy as np
import codecs

def extract_and_save_material_names():
    """提取并保存材料名称列表"""
    # 数据路径
    train_path = 'd:\\Projects\\Python_projects\\dpl\\Algorithm\\system\\complete_materials_train.csv'
    val_path = 'd:\\Projects\\Python_projects\\dpl\\Algorithm\\system\\complete_materials_val.csv'
    
    # 尝试不同的编码格式读取数据
    encodings = ['gbk', 'gb2312', 'gb18030', 'cp936', 'latin-1', 'utf-8-sig', 'iso-8859-1']
    
    # 加载训练和验证数据
    train_df = None
    val_df = None
    
    # 尝试多种编码方式
    for encoding in encodings:
        try:
            print(f"尝试使用 {encoding} 编码读取训练集...")
            train_df = pd.read_csv(train_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取训练集！")
            
            print(f"尝试使用 {encoding} 编码读取验证集...")
            val_df = pd.read_csv(val_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取验证集！")
            
            # 如果读取成功，跳出循环
            break
        except Exception as e:
            print(f"使用 {encoding} 编码读取失败: {e}")
            continue
    
    if train_df is None or val_df is None:
        raise ValueError("无法读取数据文件，请检查文件格式和编码。")
    
    # 查看数据前几行以确认读取正常
    print("训练集前几行:")
    print(train_df.iloc[:2, :5])  # 只显示前5列
    
    # 提取材料名称和对应的类别索引
    material_names = []
    class_indices = {}
    
    # 从训练集提取
    for i in range(train_df.shape[0]):
        try:
            row = train_df.iloc[i]
            name = row.iloc[0]
            idx = row.iloc[1:105].values.argmax()
            material_names.append(name)
            class_indices[name] = idx
        except Exception as e:
            print(f"处理训练集第{i}行时出错: {e}")
    
    # 从验证集提取
    for i in range(val_df.shape[0]):
        try:
            row = val_df.iloc[i]
            name = row.iloc[0]
            idx = row.iloc[1:105].values.argmax()
            material_names.append(name)
            class_indices[name] = idx
        except Exception as e:
            print(f"处理验证集第{i}行时出错: {e}")
    
    # 去重并排序
    material_names = sorted(list(set(material_names)))
    
    # 打印一些信息
    train_labels = train_df.iloc[:, 1:105].values.argmax(axis=1)
    val_labels = val_df.iloc[:, 1:105].values.argmax(axis=1)
    unique_train_labels = set(train_labels)
    unique_val_labels = set(val_labels)
    all_unique_labels = set(np.concatenate([train_labels, val_labels]))
    
    print(f"训练集中的唯一类别数量: {len(unique_train_labels)}")
    print(f"验证集中的唯一类别数量: {len(unique_val_labels)}")
    print(f"合并后的唯一类别数量: {len(all_unique_labels)}")
    print(f"材料名称数量: {len(material_names)}")
    print(f"类别索引映射数量: {len(class_indices)}")
    
    # 打印一些材料名称作为示例
    print("部分材料名称示例:")
    for i, name in enumerate(material_names[:10]):
        print(f"{i+1}. {name} (类别索引: {class_indices[name]})")
    
    # 保存材料名称列表
    os.makedirs('models', exist_ok=True)
    with open('models/material_names.pkl', 'wb') as f:
        pickle.dump(material_names, f)
    
    # 保存类别索引映射
    with open('models/class_indices.pkl', 'wb') as f:
        pickle.dump(class_indices, f)
    
    print(f"已保存 {len(material_names)} 种材料名称到 models/material_names.pkl")
    print(f"已保存类别索引映射到 models/class_indices.pkl")
    
    return material_names

if __name__ == "__main__":
    try:
        material_names = extract_and_save_material_names()
        print("材料名称提取和保存成功！")
    except Exception as e:
        print(f"材料名称提取过程中出错: {e}")
