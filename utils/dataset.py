import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
import random


class MyQueryDataset(Dataset):
    def __init__(self, dataframe, use_balance=False):
        self.raw = dataframe.values.tolist()
        self.use_balance = use_balance

        if use_balance:
            self.build_balanced_indices()

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        row = self.raw[self.indices[idx]] if self.use_balance else self.raw[idx]
        q = [int(x) for x in row[:-1]]
        label = float(row[-1])
        return torch.tensor(q, dtype=torch.long), torch.tensor(label, dtype=torch.float)

    @staticmethod
    def collate_fn(batch):
        queries, labels = zip(*batch)
        return list(queries), torch.stack(labels)

    def build_balanced_indices(self):
        element_freq = Counter()
        for row in self.raw:
            for e in set(row[:-1]):  # 只统计元素频率（去重）
                element_freq[int(e)] += 1

        # 反频率采样权重
        weights = []
        for row in self.raw:
            freq = np.mean([element_freq[int(e)] for e in row[:-1]])
            weight = 1.0 / (freq + 1e-3)
            weights.append(weight)

        weights = np.array(weights)
        weights /= weights.sum()

        # 生成平衡采样索引
        self.indices = np.random.choice(np.arange(len(self.raw)), size=len(self.raw), replace=True, p=weights)
