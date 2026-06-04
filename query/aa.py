import csv

# 输入和输出文件路径
input_file = '/workspace/ACE/overlap_queries.csv'  # 替换为你的输入CSV文件路径
output_file = 'output.csv'  # 替换为输出CSV文件路径

# 读取并转换CSV
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        if row:  # 确保行不为空
            # 第一列是除最后一列外的所有元素，第二列是最后一列
            first_column = ' '.join(row[:-1])  # 除最后一列外的元素用空格连接
            second_column = row[-1]  # 最后一列
            writer.writerow([first_column, second_column])

print(f"转换完成，输出文件已保存为 {output_file}")