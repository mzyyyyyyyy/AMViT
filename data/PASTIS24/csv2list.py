import csv

def read_csv_column_to_list(file_path, column_name):
    result_list = []
    
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            result_list.append(float(row[column_name]))
    
    return result_list

# 示例使用
file_path = './mixed_gaussian_distribution.csv'  # 替换为你的 CSV 文件路径
column_name = 'Probability'  # 替换为你想要读取的列名

result_list = read_csv_column_to_list(file_path, column_name)

print(result_list)
