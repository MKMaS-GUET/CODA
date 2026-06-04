import pandas as pd
import os

# 路径参数
input_file = '/opt/data/private/ACE/query/wiki/subset/test_regular/wiki.csv'            # 你的测试集总文件
output_dir = '/opt/data/private/ACE/query/wiki/dynamic/subset/test_regular'              # 输出目录
num_splits = 6                         # 划分数量

# 读入数据
df = pd.read_csv(input_file, header=None)

# 打乱（如不需要随机可注释掉）
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 每份的大小
split_size = len(df) // num_splits

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 划分并保存
for i in range(num_splits):
    start = i * split_size
    # 最后一份包含剩下的所有
    if i == num_splits - 1:
        split_df = df.iloc[start:]
    else:
        split_df = df.iloc[start:start + split_size]
    split_df.to_csv(f'{output_dir}/{i}.csv', index=False, header=False)

print(f"测试集已划分为 {num_splits} 份，保存在 {output_dir}/ 目录下。")
