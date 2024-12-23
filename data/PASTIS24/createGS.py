import numpy as np
import csv

def generate_mixed_gaussian_distribution(mean_centers, std_dev, value_range):
    # 初始化概率分布
    distribution = np.zeros(len(value_range))
    
    # 对每个均值中心计算高斯分布并累加到总的分布上
    for mean in mean_centers:
        gaussian = np.exp(-0.5 * ((value_range - mean) / std_dev) ** 2)
        distribution += gaussian
    
    # 归一化分布，使总和为1
    distribution /= np.sum(distribution)
    
    return distribution

def save_distribution_to_csv(distribution, output_file):
    # 将概率分布保存到CSV文件中的一列
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Probability"])  # 添加表头
        for value in distribution:
            writer.writerow([value])

# 使用示例
mean_centers = [16, 43, 76]
std_dev = 3
value_range = np.arange(0, 80)

distribution = generate_mixed_gaussian_distribution(mean_centers, std_dev, value_range)

output_file = './mixed_gaussian_distribution.csv'
save_distribution_to_csv(distribution, output_file)

print(f"Distribution saved to {output_file}")
