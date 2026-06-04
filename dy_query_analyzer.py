import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from utils.criterion import *
from utils.customized_dataset import *
from utils.arg_parser import *
from model.encoder import *
from model.ace import *
from utils.preprocess import *

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = lambda_
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class AlignNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class DomainClassifier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dim, dim // 2),
            nn.BatchNorm1d(dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    args = make_args()
    dataset = args.d
    dim = args.dim
    workload_type = args.qt
    workload_freq = args.qf
    batch_size = 10
    dev = torch.device(f'cuda:{args.dev}' if args.dev >= 0 and torch.cuda.is_available() else 'cpu')
    distill_ratio = args.r

    _, num_element = read_data(dataset)
    element_emb = torch.nn.Embedding(num_element + 10000, dim, padding_idx=0).to(dev)

    base_path = f'save_model/{dataset}_groups.pth'
    new_path = f'save_model/{dataset}_groups_new.pth'
    base_group = torch.load(base_path).to(dev)
    align_net = AlignNet(dim).to(dev)
    domain_cls = DomainClassifier(dim).to(dev)

    if os.path.exists(new_path):
        new_group = torch.load(new_path).to(dev)
        min_len = min(len(base_group), len(new_group))
        base_group = base_group[:min_len]
        new_group = new_group[:min_len]

        domain_labels = torch.cat([
            torch.zeros(min_len),
            torch.ones(min_len)
        ]).long().to(dev)
    else:
        new_group = torch.empty_like(base_group)
        domain_labels = torch.zeros(len(base_group)).long().to(dev)

    degree_path = f'{dataset}_freq.npy'
    new_data_path = f'/workspace/ACE /data/{dataset}_new.csv'
    old_degree = np.load(degree_path)
    new_df = pd.read_csv(new_data_path, header=None)
    all_ids = []
    for line in new_df[0].astype(str).fillna(''):
        tokens = [x for x in line.strip().replace('\t', ' ').replace('\u3000', ' ').replace('\xa0', ' ').split() if x.isdigit()]
        all_ids.extend(map(int, tokens))

    from collections import Counter
    new_freq = Counter(all_ids)
    max_id = max(max(new_freq.keys(), default=0), len(old_degree) - 1)
    new_degree = np.zeros(max_id + 1, dtype=np.float32)
    new_degree[:len(old_degree)] = old_degree
    for k, v in new_freq.items():
        new_degree[k] += v
    np.save(degree_path, new_degree)
    degree = torch.from_numpy(new_degree).float().to(dev)

    estimator = ACE(dim, args.cross_attn_num, args.self_attn_num, args.mlp_dep, args.mlp_dim).to(dev)
    model_path = f'save_model/{dataset}_estimator_{workload_type}_{distill_ratio}.pth'
    if Path(model_path).exists():
        estimator.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.Adam(list(estimator.parameters()) + list(align_net.parameters()), lr=1e-4)
    optimizer_cls = torch.optim.Adam(domain_cls.parameters(), lr=1e-4)

    start_time = time.time()
    for epoch in range(10):
        align_net.train()
        domain_cls.train()

        new_aligned = align_net(new_group)
        group_all = torch.cat([base_group, new_aligned], dim=0)
        group_norm = F.normalize(group_all, dim=1)

        # 1. Train domain classifier (detach GRL)
        logits = domain_cls(group_norm.detach())
        loss_domain = F.nll_loss(logits, domain_labels)
        optimizer_cls.zero_grad()
        loss_domain.backward()
        optimizer_cls.step()

        # 2. Train AlignNet + Estimator with GRL
        grl_output = GRL.apply(group_norm, min(1.0, epoch / 5))
        logits_adv = domain_cls(grl_output)
        loss_align = F.nll_loss(logits_adv, domain_labels)
        optimizer.zero_grad()
        loss_align.backward()
        optimizer.step()

        print(f"\U0001F4D8 Epoch {epoch}: Domain Loss = {loss_domain.item():.4f}, Align Loss = {loss_align.item():.4f}")

    end_time = time.time()
    print(f"⏱️ 总耗时：{end_time - start_time:.2f} 秒")

    test_path = f'/workspace/ACE /query/{dataset}/dynamic/{workload_type}/{dataset}.csv'
    test_loader = DataLoader(MyQueryDataset(pd.read_csv(test_path, header=None)), batch_size=batch_size, collate_fn=MyQueryDataset.collate_fn)

    with torch.no_grad():
        test_preds, test_truth = torch.tensor([]).to(dev), torch.tensor([]).to(dev)
        final_group = F.normalize(torch.cat([base_group, align_net(new_group)], dim=0), dim=1)
        for test_qs, test_cs in test_loader:
            test_truth = torch.cat((test_truth, test_cs.to(dev)))
            test_lst = [element_emb((q + 1).to(dev)) for q in test_qs]
            test_sel = [torch.log(degree[q] + 1e-6).to(dev) for q in test_qs]
            test_sel = pad_sequence(test_sel, True).unsqueeze(-1)
            test_lst = pad_sequence(test_lst, True)
            test_pred, _ = estimator(test_lst, final_group, test_sel)
            test_preds = torch.cat((test_preds, test_pred))

        qerr = q_error_criterion(test_truth, test_preds)
        print(f"\U0001F4CA Q-error：mean={qerr.mean():.3f}, median={qerr.median():.3f}, 95%={qerr.quantile(0.95):.3f}, 99%={qerr.quantile(0.99):.3f}, max={qerr.max():.3f}")
        align_save_path = f'save_model/{dataset}_{workload_type}_align_net.pth'
        torch.save(align_net.state_dict(), align_save_path)
        print(f"✅ AlignNet 已保存到 {align_save_path}")