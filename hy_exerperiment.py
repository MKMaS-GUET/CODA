# 网格搜索实验主脚本：训练 + 生成 + 测试

import os
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from utils.arg_parser import make_args
from utils.preprocess import seed_random, read_data
from utils.criterion import q_error_criterion
from utils.customized_dataset import MyQueryDataset, SequentialDataset
from model.encoder import Aggregator, PostDistillation, ModelRegularizer
from model.ace import ACE

# 参数范围定义
qt_list = ["subset", "superset", "overleap"]
r_list = [0.001, 0.002, 0.004]
n_distill_list = [2, 4, 6, 8]
n_cross_list = [2, 4, 6]
n_self_list = [4, 8, 12]

# 预处理：构建group embedding

def build_group_embedding(dataset, dim, distill_ratio, batch_size, device):
    print("Step 1️⃣: 正在生成group embedding...")
    data, num_element = read_data(dataset)
    element_emb = torch.nn.Embedding(num_element + 1, dim, padding_idx=0).requires_grad_(False)

    agg_model = Aggregator(dim).to(device)
    dis_model = PostDistillation(4, dim, dim, False).to(device)
    agg_model.load_state_dict(torch.load(f'save_model/{dataset}_aggregation.pth'))
    dis_model.load_state_dict(torch.load(f'save_model/{dataset}_distillation.pth'))

    dataset_obj = SequentialDataset(data, 1, 8, num_element)
    loader = DataLoader(dataset_obj, batch_size=batch_size, collate_fn=SequentialDataset.collate_fn)
    group_embs = torch.tensor([]).to(device)

    with torch.no_grad():
        for data, pos_idx, neg_idx in loader:
            data = pad_sequence(data, True)
            element_embs = element_emb(data).to(device)
            masks = torch.sign(torch.sum(torch.abs(element_embs), -1)).unsqueeze(-1).expand(-1, -1, dim)
            set_embs, _ = agg_model(element_embs, masks)
            idx = torch.randperm(set_embs.size(0))[:max(int(set_embs.size(0) * distill_ratio), 2)]
            groups = set_embs[idx].detach()
            groups = dis_model(set_embs.unsqueeze(0), groups)
            group_embs = torch.cat((group_embs, groups))

    torch.save(group_embs, f'save_model/{dataset}_groups.pth')
    print("✅ Group embedding 已保存")

# 载入数据

def load_workload(dataset, qt, qf, split, batch_size):
    if split == 'test':
        path = f'/workspace/ACE /query/{dataset}/{qt}/test_{qf}/{dataset}.csv'
    else:
        path = f'/workspace/ACE /query/{dataset}/{qt}/{split}_workload.csv'
    data = pd.read_csv(path, header=None)
    dataset = MyQueryDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=MyQueryDataset.collate_fn)

# 计算模型大小

def compute_model_size(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / 1024 / 1024  # MB

# 主逻辑：训练 + 测试 + 记录结果

def run():
    args = make_args()
    args.d = "wiki"
    device = torch.device(f'cuda:{args.dev}' if args.dev >= 0 else 'cpu')
    seed_random(args.s)
    degree = torch.from_numpy(np.load(f'{args.d}_freq.npy')).to(device)

    results = []
    for qt in qt_list:
        for r in r_list:
            build_group_embedding(args.d, args.dim, r, args.db, device)
            for n_distill in n_distill_list:
                for n_cross in n_cross_list:
                    for n_self in n_self_list:

                        element_emb, group_embs = prepare_embeddings(args.d, args.dim, device)
                        group_embs = group_embs[:int(r * group_embs.size(0))]
                        test_loader = load_workload(args.d, qt, args.qf, 'test', args.qb)

                        model = ACE(args.dim, n_cross, n_self, n_distill, args.mlp_dim).to(device)
                        model.load_state_dict(torch.load(f'save_model/{args.d}_estimator_{qt}_{r}.pth'))

                        # 推理
                        model.eval()
                        preds, truth, latency = [], [], []
                        with torch.no_grad():
                            for qs, cs in test_loader:
                                truth.append(cs)
                                lst = [element_emb(q + 1) for q in qs]
                                sel = [torch.log(degree[q] + 1e-6) for q in qs]
                                sel = pad_sequence(sel, True).unsqueeze(-1).to(device)
                                lst = pad_sequence(lst, True).to(device)
                                t1 = time.perf_counter()
                                pred, _ = model(lst, group_embs, sel)
                                latency.append((time.perf_counter() - t1) * 1000)
                                preds.append(pred.cpu())

                        qerr = q_error_criterion(torch.cat(truth), torch.cat(preds))
                        results.append({
                            "qt": qt, "r": r, "n_distill": n_distill, "n_cross": n_cross, "n_self": n_self,
                            "mean_qerr": qerr.mean().item(),
                            "latency_ms": round(np.mean(latency), 3),
                            "model_size_mb": round(compute_model_size(model), 3),
                            "data_size_mb": round(group_embs.numel() * 4 / 1024 / 1024, 3)
                        })
                        print(f"✅ {qt} r={r} d={n_distill} c={n_cross} s={n_self} qerr={qerr.mean().item():.3f}")

    df = pd.DataFrame(results)
    df.to_csv("gridsearch_result.csv", index=False)
    print("✅ 所有实验结果保存在 gridsearch_result.csv")


def prepare_embeddings(dataset, dim, device):
    _, num_element = read_data(dataset)
    emb = torch.nn.Embedding(num_element + 1, dim, padding_idx=0).requires_grad_(False)
    group_model = torch.load(f'save_model/{dataset}_groups.pth')
    group_embs = group_model.weight.detach().to(device) if hasattr(group_model, 'weight') else group_model.detach().to(device)
    return emb, group_embs

if __name__ == '__main__':
    run()
