import os
import re
import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from utils.preprocess import *
from utils.customized_dataset import *
from model.encoder import *
from model.ace import *
from utils.arg_parser import make_args

def load_and_extend_embedding(element_emb, max_id, dim):
    if max_id < element_emb.num_embeddings:
        return element_emb
    new_emb = torch.nn.Embedding(max_id + 1, dim, padding_idx=0)
    with torch.no_grad():
        new_emb.weight[:element_emb.num_embeddings] = element_emb.weight.data
        nn.init.normal_(new_emb.weight[element_emb.num_embeddings:], std=0.01)
    new_emb.requires_grad_(False)
    return new_emb

def load_new_sets_as_strings(new_path):
    df = pd.read_csv(new_path, header=None)
    clean_sets = []

    for i, line in enumerate(df[0]):
        if pd.isna(line):
            continue
        line = str(line).strip().replace('\u3000', ' ').replace('\t', ' ').replace('\xa0', ' ')
        tokens = re.split(r'[,\s]+', line)
        filtered = [x for x in tokens if x.isdigit()]
        if not filtered:
            print(f"⚠️ 跳过非法或空行：第 {i+1} 行 内容: {line}")
            continue
        clean_line = ' '.join(filtered)
        clean_sets.append(clean_line)

    print(f"✅ 加载集合数: {len(clean_sets)}")
    return clean_sets

if __name__ == '__main__':
    args = make_args()
    dataset = args.d
    dim = args.dim
    distill_ratio = args.r
    device = torch.device(f'cuda:{args.dev}' if args.dev >= 0 and torch.cuda.is_available() else 'cpu')

    print("🔁 加载 base.csv 获取元素数")
    base_path = f'/workspace/ACE /data/{dataset}.csv'
    base_df = pd.read_csv(base_path, header=None)
    num_element = int(base_df.iloc[0, 0])

    print("📥 加载新集合 new_data.csv")
    new_path = f'/workspace/ACE /data/{dataset}_new.csv'
    raw_sets = load_new_sets_as_strings(new_path)
    new_sets = []
    for i, s in enumerate(raw_sets):
        fixed = re.sub(r'\s+', ' ', s.strip())
        if '' in fixed.split(' '):
            print(f"❌ 第 {i+1} 行仍含空字符串！内容: '{fixed}'")
        new_sets.append(fixed)
    all_ids = [int(x) for s in new_sets for x in s.strip().split()]
    max_new_id = max(all_ids)
    total_elements = max(max_new_id, num_element)

    print("🔧 初始化 element embedding（自动扩容）")
    element_emb = torch.nn.Embedding(total_elements + 1, dim, padding_idx=0).requires_grad_(False)
    element_emb = load_and_extend_embedding(element_emb, total_elements, dim)
    element_emb = element_emb.to(device)

    print("📦 加载 aggregator 和 distiller 模型")
    agg_model = Aggregator(dim).to(device)
    agg_model.load_state_dict(torch.load(f'save_model/{dataset}_aggregation.pth', map_location=device))
    agg_model.eval()

    dis_model = PostDistillation(args.dis_dep, dim, dim, False).to(device)
    dis_model.load_state_dict(torch.load(f'save_model/{dataset}_distillation.pth', map_location=device))
    dis_model.eval()

    print("📤 构建数据集并蒸馏")
    seq_dataset = SequentialDataset(pd.Series(new_sets), 1, args.neg, total_elements)
    loader = DataLoader(seq_dataset, batch_size=args.qb, collate_fn=SequentialDataset.collate_fn)
    group_embs = []
    with torch.no_grad():
        for batch_id, (data_tensor, _, _) in enumerate(loader):
            data_tensor = pad_sequence(data_tensor, True).to(device)
            emb = element_emb(data_tensor)
            masks = torch.sign(torch.sum(torch.abs(emb), -1)).unsqueeze(-1).expand(-1, -1, dim)
            set_embs, _ = agg_model(emb, masks)

            num_sets = set_embs.size(0)
            sample_size = max(int(num_sets * distill_ratio), 2)

            print(f"\n🧪 Batch {batch_id}:")
            print(f"   🔹 Set embedding 数量: {num_sets}")
            print(f"   🔹 目标 sample_size: {sample_size}")

            if num_sets < 2:
                print("   ⚠️ 当前 batch 中的 set 太少，跳过该 batch")
                continue

            # 加上保护，防止越界
            sample_size = min(sample_size, num_sets)
            group_idx = torch.randperm(num_sets)[:sample_size]
            print(f"   ✅ 实际采样数量: {len(group_idx)}")



            groups = set_embs[group_idx].detach()
            print(f"set_embs device: {set_embs.device}, groups device: {groups.device}")

            distilled = dis_model(set_embs.unsqueeze(0), groups)


            group_embs.append(distilled.cpu())

