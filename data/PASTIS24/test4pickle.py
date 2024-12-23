import numpy as np

# 假设你的 .npy 文件路径是 'path_to_file.npy'
file_path = "D:\\PASTIS\\All_PASTIS_Data\\ANNOTATIONS_PART\\TARGET_10002.npy"

# 读取 .npy 文件
data = np.load(file_path)

# 打印数据的 shape
print(f"The shape of the .npy file data is: {data.shape}")
