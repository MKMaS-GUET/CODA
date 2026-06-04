import csv

# 输入和输出文件路
input_file = '/workspace/ACE /new_sets.csv'  # 替换为你的输入CSV文件路径
output_file = '/workspace/ACE /data/gn_ne.csv'  # 替换为输出CSV文件路径

# 读取并转换CSV
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        if row:  # 确保行不为空
            # 将每行所有元素用空格连接成单行，并写入
            single_row = ' '.join(row)
            writer.writerow([single_row])

print(f"转换完成，输出文件已保存为 {output_file}")