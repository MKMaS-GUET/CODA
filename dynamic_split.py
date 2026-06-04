import os
import random
import pandas as pd
from pathlib import Path

# ===== 配置部分 =====
dataset_name = 'wiki'
input_file = '/opt/data/private/ACE/data/wiki.csv'  # 每行是一个 set（用 , 分隔元素）
output_dir = Path(f'/opt/data/private/ACE/data/dynamic/{dataset_name}')
output_dir.mkdir(parents=True, exist_ok=True)

# ===== 加载全集数据 =====
with open(input_file, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

# ===== 自动统计元素全集大小 =====
all_elements = set()
for line in lines:
    elements = line.split(',')
    all_elements.update(elements)
element_num = len(all_elements)  # 自动统计全集元素数量

random.shuffle(lines)
total = len(lines)
assert total >= 10, "数据太少，不足以划分"

# ===== 划分数量 =====
num_base = int(total * 0.5)
num_update = int(total * 0.1)
remaining = total - num_base
num_parts = remaining // num_update
if remaining % num_update != 0:
    num_parts += 1

# ===== 写 base.csv =====
with open(output_dir / 'base.csv', 'w') as f:
    f.write(f"{element_num}\n")
    for line in lines[:num_base]:
        f.write(line + '\n')

# ===== 写 {i}.csv =====
start = num_base
for i in range(num_parts):
    end = min(start + num_update, total)
    subset = lines[start:end]
    if not subset:
        break
    is_delete = random.choice([0, 1])
    with open(output_dir / f'{i}.csv', 'w') as f:
        f.write(f"{is_delete}\n")
        for line in subset:
            f.write(line + '\n')
    start = end

print(f"✅ 数据划分完成，结果保存在：{output_dir}")
