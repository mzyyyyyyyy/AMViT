import os
import numpy as np
from collections import Counter
from tqdm import tqdm

def count_pixel_values_in_npy_files(folder_path):
    # 初始化一个 Counter 对象，用于统计所有文件中不同像素值的个数
    total_pixel_counts = Counter()

    # 遍历文件夹中所有的 .npy 文件
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".npy"):
            file_path = os.path.join(folder_path, filename)

            # 加载 .npy 文件
            data = np.load(file_path)
            
            # 使用 Counter 统计当前文件中不同像素值的个数
            pixel_counts = Counter(data.flatten())
            
            # 将当前文件的统计结果添加到总的统计结果中
            total_pixel_counts.update(pixel_counts)

    return total_pixel_counts

# 示例使用
folder_path = 'D:\\PASTIS\\ALL_GXData\\ANNOTATIONS'  # 替换为你的文件夹路径
pixel_counts = count_pixel_values_in_npy_files(folder_path)

# 打印统计结果
for pixel_value, count in pixel_counts.items():
    print(f"Pixel value {pixel_value}: {count} pixels")
