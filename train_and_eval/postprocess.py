import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取Excel表格数据
file_path = "C:\\Users\\mzy20\\Desktop\\研究进展汇报\\组会\\10月论文\\混淆矩阵.xlsx"  # 请将此路径替换为实际的Excel文件路径
df = pd.read_excel(file_path, header=None, usecols="C:M", skiprows=50, nrows=11)

# 设置行名称和列名称
df.index = ['CF', 'MP', 'T', 'EU', 'EGU', 'SBS', 'MW', 'SB', 'FT', 'ERMF', 'RHS']  # 替换为实际的行名称
df.columns = ['CF', 'MP', 'T', 'EU', 'EGU', 'SBS', 'MW', 'SB', 'FT', 'ERMF', 'RHS']  # 替换为实际的列名称

df = df.round(4)

# 将数据转化为热力图
plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=True, fmt=".4f", cmap='viridis')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
