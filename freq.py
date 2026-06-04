import numpy as np

# 文件路径
filename = '/workspace/ACE /data/tweet.csv'

# 读文件
with open(filename, 'r') as f:
    lines = f.readlines()

num_elements = int(lines[0].strip())
degrees = np.zeros(num_elements, dtype=np.int32)

# 每一行统计元素度数
for line in lines[1:]:
    # 每行的元素编号
    elements = list(map(int, line.strip().split()))
    for e in elements:
        degrees[e] += 1

# 保存到npy
np.save('tweet_freq.npy', degrees)
print(f'已保存度数到 sets_freq.npy, 共 {num_elements} 个元素')
