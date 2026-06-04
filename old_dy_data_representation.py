import os
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils.preprocess import *
from utils.customized_dataset import *
from utils.arg_parser import *
from model.encoder import *
from model.ace import *

if __name__ == '__main__':
    args = make_args()
    dataset = args.d
    dim = args.dim
    batch_size = args.qb
    distill_ratio = args.r
    dev = 'cpu' if args.dev < 0 else f'cuda:{args.dev}'
    print(args)

    seed_random(args.s)

    # ===== 1. 加载 base 数据 =====
    base_path = f'/workspace/ACE /data/{dataset}.csv'
    data_base = pd.read_csv(base_path, header=None)
    num_element_old = int(data_base.iloc[0, 0])
    data_base = data_base[1:].reset_index(drop=True)

    # ===== 2. 加载新数据并转换为字符串序列 =====
    new_path = f'/workspace/ACE /data/{dataset}_new.csv'
    data_new = pd.read_csv(new_path, header=None)
    is_delete = int(data_new.iloc[0, 0])
    data_new = data_new[1:].reset_index(drop=True)

    sequences = data_new[0].astype(str).tolist()
    sequences = np.array(sequences, dtype=object)

    # ===== 3. 加载旧的 group embedding =====
    save_dir = f'/workspace/ACE /save_model/'
    os.makedirs(save_dir, exist_ok=True)
    group_path = f'{save_dir}/{dataset}_groups.pth'
    if os.path.exists(group_path):
        old_group_emb = torch.load(group_path)
        print(f"📦 旧 group embedding 加载成功: {old_group_emb.shape}")
    else:
        old_group_emb = torch.tensor([])
        print(f"⚠️ 未找到旧 group embedding，将创建新文件")

    # ===== 4. 构建模型并加载预训练参数 =====
    element_emb = torch.nn.Embedding(num_element_old + 1, dim, padding_idx=0).to(dev).requires_grad_(False)

    agg_model = Aggregator(dim).to(dev)
    agg_model.load_state_dict(torch.load(f'{save_dir}/{dataset}_aggregation.pth', map_location=dev))
    agg_model.eval()

    dis_model = PostDistillation(args.dis_dep, dim, dim, False).to(dev)
    dis_model.load_state_dict(torch.load(f'{save_dir}/{dataset}_distillation.pth', map_location=dev))
    dis_model.eval()

    # ===== 5. 构建数据集和 Loader =====
    new_dataset = SequentialDataset(sequences, 1, args.neg, num_element_old)
    new_loader = DataLoader(new_dataset, batch_size, collate_fn=SequentialDataset.collate_fn)
    print(f"📊 新数据 batch 数: {len(new_loader)}")

    # ===== 6. 提取新 group embedding =====
    new_group_emb = torch.tensor([])

    with torch.no_grad():
        for data, pos_idx, neg_idx in new_loader:
            data = pad_sequence(data, True).to(dev)
            pos_idx = pos_idx.to(dev)
            neg_idx = neg_idx.to(dev)

            element_embs = element_emb(data)
            masks = torch.sign(torch.sum(torch.abs(element_embs), -1)).unsqueeze(-1)
            masks = masks.expand(-1, -1, element_embs.size(-1))

            set_embs, _ = agg_model(element_embs, masks)
            num_samples = set_embs.size(0)

            if num_samples < 2:
                print(f"⚠️ 当前 batch 样本太少（{num_samples}），跳过")
                continue

            num_select = min(num_samples, max(int(num_samples * distill_ratio), 2))
            group_idx = torch.randperm(num_samples)[:num_select].to(set_embs.device)

            # 安全检查（防越界）
            assert torch.all(group_idx < num_samples), f"索引越界：group_idx={group_idx}, num_samples={num_samples}"

            selected_groups = set_embs[group_idx].detach()
            distilled = dis_model(set_embs.unsqueeze(0), selected_groups)

            new_group_emb = torch.cat([new_group_emb, distilled.cpu()], dim=0)

    print(f"✅ 新 group embedding 提取完成: {new_group_emb.shape}")

    # ===== 7. 拼接并保存 =====
    if old_group_emb.nelement() > 0:
        final_group_emb = torch.cat([old_group_emb, new_group_emb], dim=0)
    else:
        final_group_emb = new_group_emb

    torch.save(final_group_emb, group_path)
    print(f"💾 已保存更新后的 group embedding: {final_group_emb.shape} => {group_path}")
    print("🎉 Group Embedding 增量更新完成")
