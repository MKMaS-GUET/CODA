from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class ObjKeyDataset(Dataset):
    def __init__(self, adj_matrix, num_neg_samples, device='cpu'):
        self.adj = adj_matrix.transpose()
        self.num_edge = adj_matrix.nnz
        self.num_key, self.num_obj = adj_matrix.shape
        self.num_negs = num_neg_samples
        self.src, self.dst = self.adj.nonzero()
        self.src = torch.from_numpy(self.src)
        self.dst = torch.from_numpy(self.dst)
        self.device = device
        self.key_tensor = torch.arange(self.num_key).to(device)
        
    
    def __len__(self):
        return self.num_edge
    

    def __getitem__(self, idx):
        # idx: the index of the selected edge
        # 1. soft negative sampling: sample nodes from the rest nodes
        # neg_sample_idx = torch.from_numpy(np.random.choice(np.setdiff1d(np.arange(0, self.num_key), np.array([self.dst[idx]])), size=(self.num_negs, )))
        neg_sample_idx = torch.randint(0, self.num_key, size=(self.num_negs, )).to(self.device)
        
        # 2. hard negative sampling: sample nodes from the unconnected nodes
        # neg_sample_idx = torch.from_numpy(np.random.choice(np.setdiff1d(np.arange(0, self.num_key), self.adj[self.src[idx]].nonzero()[1]), size=(self.num_negs, ))).to(self.device)

        # connected_dst = torch.from_numpy(self.adj[self.src[idx]].nonzero()[1]).to(self.device)
        # diff = torch.masked_select(self.key_tensor, torch.isin(self.key_tensor, connected_dst, invert=True))
        # neg_idx = torch.randint(0, len(diff), size=(self.num_negs, ))
        # neg_sample_idx = diff[neg_idx]
        
        return self.src[idx].long().to(self.device), self.dst[idx].long().to(self.device), neg_sample_idx.long()


    @staticmethod
    def collate_fn(data):
        idx_obj = torch.stack([_[0] for _ in data], dim=0)
        pos_idx_key = torch.stack([_[1] for _ in data], dim=0)
        neg_idx_key = torch.stack([_[2] for _ in data], dim=0)
        return idx_obj, pos_idx_key, neg_idx_key


class KeyObjDataset(Dataset):
    def __init__(self, adj_matrix, num_neg_samples, device):
        adj_mat_coo = adj_matrix.tocoo()
        self.adj = torch.sparse_coo_tensor(torch.LongTensor(np.vstack((adj_mat_coo.row, adj_mat_coo.col))), torch.FloatTensor(adj_mat_coo.data), adj_mat_coo.shape).to(device)
        self.num_edge = adj_matrix.nnz
        self.num_key, self.num_obj = adj_matrix.shape
        self.num_negs = num_neg_samples
        self.src, self.dst = adj_matrix.nonzero()
        self.src = torch.from_numpy(self.src)
        self.dst = torch.from_numpy(self.dst)
        self.device = device
        self.obj_tensor = torch.arange(self.num_obj).to(device)
        
    
    def __len__(self):
        return self.num_edge
    

    def __getitem__(self, idx):
        # idx: the index of the selected edge
        # 1. soft negative sampling: sample nodes from the rest nodes
        # neg_sample_idx = torch.from_numpy(np.random.choice(np.setdiff1d(np.arange(0, self.num_key), np.array([self.dst[idx]])), size=(self.num_negs, )))
        neg_sample_idx = torch.randint(0, self.num_key, size=(self.num_negs, )).to(self.device)
        
        # 2. hard negative sampling: sample nodes from the unconnected nodes
        # neg_sample_idx = torch.from_numpy(np.random.choice(np.setdiff1d(np.arange(0, self.num_key), self.adj[self.src[idx]].nonzero()[1]), size=(self.num_negs, ))).to(self.device)

        # t2 = torch.nonzero(self.adj[self.src[idx]].to_dense()).squeeze(1).to(self.device)
        # diff = torch.masked_select(self.obj_tensor, torch.isin(self.obj_tensor, t2, invert=True))
        # neg_idx = torch.randint(0, len(diff), size=(self.num_negs, ))
        # neg_sample_idx = diff[neg_idx]
        
        return self.src[idx].long().to(self.device), self.dst[idx].long().to(self.device), neg_sample_idx.long()


    @staticmethod
    def collate_fn(data):
        idx_key = torch.stack([_[0] for _ in data], dim=0)
        pos_idx_obj = torch.stack([_[1] for _ in data], dim=0)
        neg_idx_obj = torch.stack([_[2] for _ in data], dim=0)
        return idx_key, pos_idx_obj, neg_idx_obj
    

class MyQueryDataset(Dataset):
    def __init__(self, query_df):
        super(MyQueryDataset, self).__init__()
        self.num = query_df.shape[0]
        self.queries = query_df

    
    def __len__(self):
        return self.num
    

    def __getitem__(self, idx):
        card = torch.tensor(self.queries[1][idx]).unsqueeze(-1)
        keys = torch.tensor([int(k) for k in str(self.queries[0][idx]).split(' ')])

        return keys, card
    

    def collate_fn(data):
        keys = [_[0] for _ in data]
        cards = torch.stack([_[1] for _ in data], dim=0)

        return keys, cards

class SequentialDataset(Dataset):
    def __init__(self, data, num_pos, num_neg, num_element):
        super().__init__()
        self.num = data.shape[0]
        self.data = data
        self.num_poss = num_pos
        self.num_negs = num_neg
        self.num_element = num_element

    def __len__(self):
        return self.num
    def __getitem__(self, idx):
        raw_line = self.data[idx]  # ✅ 修复索引方式

        tokens = [e for e in raw_line.split(' ') if e.strip().isdigit()]
        if len(tokens) == 0:
            print(f"⚠️ 第 {idx + 1} 行是空集合或无合法元素，原始内容：'{raw_line}'")
            tokens = ['1']

        sets = torch.tensor([int(e) + 1 for e in tokens])
        pos_idx = torch.randint(0, sets.size(0), size=(self.num_poss,))
        pos_idx %= sets.size(0)
        pos_samples_idx = sets[pos_idx]
        neg_samples_idx = torch.randint(1, self.num_element + 1, size=(self.num_negs,))
        return sets, pos_samples_idx, neg_samples_idx


    def collate_fn(data):
        sets = [_[0] for _ in data]
        pos_samples_idx = torch.stack([_[1] for _ in data], dim=0)
        neg_samples_idx = torch.stack([_[2] for _ in data], dim=0)
        return sets, pos_samples_idx, neg_samples_idx
