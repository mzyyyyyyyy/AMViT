import os
import csv
import math
import random

def split_filenames_into_csv(folder_path, output_folder, num_splits=5):
    # 获取文件夹下所有的文件名
    filenames = os.listdir(folder_path)
    
    # 打乱文件名顺序
    random.shuffle(filenames)
    
    # 总文件数
    total_files = len(filenames)
    
    # 计算每份应该包含的文件数
    files_per_split = math.ceil(total_files / num_splits)
    
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 分割文件名并写入CSV文件
    for i in range(num_splits):
        start_index = i * files_per_split
        end_index = min((i + 1) * files_per_split, total_files)
        split_filenames = filenames[start_index:end_index]
        
        # 准备CSV文件路径
        csv_filename = os.path.join(output_folder, f'split_{i + 1}.csv')
        
        # 写入CSV文件
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for filename in split_filenames:
                writer.writerow([f"pickle32x32/{filename}"])
                
        # print(f"Generated {csv_filename} with {len(split_filenames)} filenames.")

# 使用示例
folder_path = 'D:\\DeepSatModels_Data\\data\\GXData\\pickle32x32'  # 替换为包含文件的文件夹路径
output_folder = 'D:\\DeepSatModels_Data\\data\\GXData\\fold-paths'  # 替换为存储CSV文件的文件夹路径
split_filenames_into_csv(folder_path, output_folder)
