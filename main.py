import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from utils.preprocess import seed_random, read_data
from utils.criterion import weight_q_error, q_error_criterion
from utils.customized_dataset import MyQueryDataset
from utils.arg_parser import make_args
from model.ace import ACE
from model.encoder import ModelRegularizer

# Contrastive loss
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    similarity_matrix = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(batch_size).to(z1.device)
    return F.cross_entropy(similarity_matrix, labels)

# Augment queries with masking and shuffling
def augment_query_tensor(lst, mask_ratio=0.3, shuffle_ratio=0.3):
    B, L, D = lst.size()
    lst_aug = lst.clone()
    for i in range(B):
        mask_len = int(L * mask_ratio)
        if mask_len > 0:
            mask_idx = torch.randperm(L)[:mask_len]
            lst_aug[i, mask_idx] = 0
        shuffle_len = int(L * shuffle_ratio)
        if shuffle_len > 1:
            shuffle_idx = torch.randperm(shuffle_len)
            permuted = lst_aug[i, :shuffle_len][shuffle_idx]
            lst_aug[i, :shuffle_len] = permuted
    return lst_aug

def prepare_embeddings(dataset, dim, device):
    _, num_element = read_data(dataset)
    element_emb = nn.Embedding(num_element + 1, dim, padding_idx=0).requires_grad_(False)
    group_embs = torch.load(f'save_model/{dataset}_groups.pth').detach().to(device).requires_grad_(False)
    return element_emb, group_embs

def load_workload(dataset, workload_type, workload_freq, split, batch_size, nrows=None):
    if split in ['train', 'val']:
        path = f'./query/{dataset}/{workload_type}/{split}_workload.csv'
    else:
        path = f'./query/{dataset}/{workload_type}/{split}_{workload_freq}/{dataset}.csv'
    print(f"Loading from: {path}")
    data = pd.read_csv(path, header=None, nrows=nrows)
    dataset = MyQueryDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), collate_fn=MyQueryDataset.collate_fn)

def train(estimator, element_emb, group_embs, degree, train_loader, val_loader, optimizer, regularizer, args, device):
    best_loss = float('inf')
    for epoch in range(args.qe):
        estimator.train()
        for qs, truth in train_loader:
            truth = truth.float().to(device)
            lst = [element_emb(q + 1) for q in qs]
            sel = [torch.log(degree[q]) for q in qs]
            sel = pad_sequence(sel, True).unsqueeze(-1).to(device)
            lst = pad_sequence(lst, True).to(device)

            pred, z1 = estimator(lst, group_embs, sel)
            lst_aug = augment_query_tensor(lst)
            _, z2 = estimator(lst_aug, group_embs, sel)

            pred_loss = weight_q_error(truth, pred).sum()
            cl_loss = contrastive_loss(z1, z2)
            total_loss = pred_loss + 0.1 * cl_loss + regularizer(estimator)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if epoch >= 10:
            estimator.eval()
            with torch.no_grad():
                val_preds, val_truth = [], []
                for val_qs, val_cs in val_loader:
                    val_truth.append(val_cs)
                    val_lst = [element_emb(q + 1) for q in val_qs]
                    val_sel = [torch.log(degree[q]) for q in val_qs]
                    val_sel = pad_sequence(val_sel, True).unsqueeze(-1).to(device)
                    val_lst = pad_sequence(val_lst, True).to(device)
                    val_pred, _ = estimator(val_lst, group_embs, val_sel)
                    val_preds.append(val_pred.cpu())
                val_loss = q_error_criterion(torch.cat(val_truth), torch.cat(val_preds)).mean()
                print(f"Epoch {epoch}: Val Loss = {val_loss.item():.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(estimator.state_dict(), f'save_model/{args.d}_estimator.pth')

def test(estimator, element_emb, group_embs, degree, test_loader, device):
    estimator.eval()
    preds, truth = [], []
    times = []
    with torch.no_grad():
        for test_qs, test_cs in test_loader:
            start = time.perf_counter()
            truth.append(test_cs)
            test_lst = [element_emb(q + 1) for q in test_qs]
            test_sel = [torch.log(degree[q]) for q in test_qs]
            test_sel = pad_sequence(test_sel, True).unsqueeze(-1).to(device)
            test_lst = pad_sequence(test_lst, True).to(device)
            test_pred, _ = estimator(test_lst, group_embs, test_sel)
            preds.append(test_pred.cpu())
            times.append((time.perf_counter() - start) * 1000)
    qerr = q_error_criterion(torch.cat(truth), torch.cat(preds))
    print("Test Results:")
    print("Mean: %.3f, Median: %.3f, 95%%: %.3f, 99%%: %.3f, Max: %.3f, Time(ms): %.3f" % (
        qerr.mean().item(), qerr.median().item(),
        qerr.quantile(0.95).item(), qerr.quantile(0.99).item(),
        qerr.max().item(), np.mean(times)
    ))

def main():
    args = make_args()
    device = torch.device(f'cuda:{args.dev}' if args.dev >= 0 else 'cpu')
    seed_random(args.s)

    element_emb, group_embs = prepare_embeddings(args.d, args.dim, device)
    group_embs = group_embs[:int(0.7 * group_embs.size(0))]

    estimator = ACE(args.dim, args.cross_attn_num, args.self_attn_num, args.mlp_dep, args.mlp_dim).to(device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=args.ql)
    regularizer = ModelRegularizer(args.qr)
    degree = torch.from_numpy(np.load(f'{args.d}_freq.npy')).to(device)

    train_loader = load_workload(args.d, args.qt, args.qf, 'train', args.qb)
    val_loader = load_workload(args.d, args.qt, args.qf, 'val', args.qb)
    test_loader = load_workload(args.d, args.qt, args.qf, 'test', args.qb)

    if args.m == 'train':
        train(estimator, element_emb, group_embs, degree, train_loader, val_loader, optimizer, regularizer, args, device)
    else:
        estimator.load_state_dict(torch.load(f'save_model/{args.d}_estimator.pth'))
        test(estimator, element_emb, group_embs, degree, test_loader, device)

if __name__ == '__main__':
    main()
















import torch
import torch.nn as nn
import torch.nn.functional as F
from model.att_modules import *
from model.encoder import *  # 确保其中包含 ProjectionHead 类

class ACE(nn.Module):
    def __init__(self, dim, cross_att_layers, self_att_layers, mlp_layers, mlp_dim, data_num=None, agg_model=None):
        super().__init__()

        self.analyzer1 = nn.ModuleList([
            nn.ModuleList([PostNormAttention(dim, dim, hidden_dim=dim), PostNormFFN(dim)])
            for _ in range(cross_att_layers)
        ])
        self.analyzer2 = nn.ModuleList([
            nn.ModuleList([PostNormAttention(dim, dim, hidden_dim=dim), PostNormFFN(dim)])
            for _ in range(self_att_layers)
        ])

        self.mean_pool = nn.AdaptiveAvgPool2d((1, dim + 1))
        self.pool = Pooling(dim + 1, 1)
        self.mlp = BasicMLP([dim + 1] + [mlp_dim] * (mlp_layers - 1) + [1])

        self.projection_head = ProjectionHead(dim + 1)  # 新增：用于对比学习的投影头

    def forward(self, q, x, sel):
        q = torch.clamp(q, -50, 50)
        x = torch.clamp(x, -50, 50)
        sel = torch.clamp(sel, -50, 50)

        x = x.repeat(q.size(0), 1, 1)
        q_mask = torch.sign(torch.sum(torch.abs(q), dim=-1))  # [B, L]

        for attn, ffn in self.analyzer1:
            q = torch.nan_to_num(q, nan=0.0, posinf=1e3, neginf=-1e3)
            q = attn(q, x)
            q = ffn(q)

        for attn, ffn in self.analyzer2:
            q = torch.nan_to_num(q, nan=0.0, posinf=1e3, neginf=-1e3)
            q = attn(q)
            q = ffn(q)

        mask = q_mask.unsqueeze(-1).expand(-1, -1, q.size(-1))
        q = q * mask
        q = torch.cat([q, sel], dim=-1)
        q = self.pool(q, q_mask)

        q = torch.nan_to_num(q, nan=0.0, posinf=1e3, neginf=-1e3)
        res = self.mlp(q)
        proj = self.projection_head(q)  # 新增：投影向量输出

        return res, proj  # 同时返回原始回归输出和对比学习表示


class Pooling(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.mab = Attention(dim, dim, dim, heads)
        self.pooling_emb = nn.Parameter(torch.zeros(1, dim))
        self.pool = nn.AdaptiveAvgPool2d((1, dim))
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.ffn = BasicMLP([dim, 512, dim])
        nn.init.trunc_normal_(self.pooling_emb, std=0.02)

    def forward(self, x, mask=None):
        x = torch.clamp(x, -50, 50)
        out = self.pool(x)
        out = self.norm1(out + self.mab(out, x, k_mask=mask))
        out = self.norm2(out + self.ffn(out))
        return torch.nan_to_num(out.squeeze(1), nan=0.0, posinf=1e3, neginf=-1e3) 
                                





import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.preprocess import seed_random, read_data
from utils.criterion import weight_q_error, q_error_criterion
from utils.customized_dataset import MyQueryDataset
from utils.arg_parser import make_args
from model.ace import ACE  # 对比学习版也在 model/ace.py 中，根据 args.qt 判断
from model.encoder import ModelRegularizer


def prepare_embeddings(dataset, dim, device):
    _, num_element = read_data(dataset)
    element_emb = nn.Embedding(num_element + 1, dim, padding_idx=0).requires_grad_(False)
    group_embs = torch.load(f'save_model/{dataset}_groups.pth').detach().to(device).requires_grad_(False)
    return element_emb, group_embs


def load_workload(dataset, workload_type, workload_freq, split, batch_size, nrows=None):
    if split in ['train', 'val']:
        path = f'./query/{dataset}/{workload_type}/{split}_workload.csv'
    else:
        path = f'./query/{dataset}/{workload_type}/{split}_{workload_freq}/{dataset}.csv'
    print(f"Loading from: {path}")
    data = pd.read_csv(path, header=None, nrows=nrows)
    dataset = MyQueryDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), collate_fn=MyQueryDataset.collate_fn)


import torch
import torch.nn.functional as F

def pairwise_jaccard(qs):
    n = len(qs)
    jac = torch.zeros((n, n), device='cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(n):
        set_i = set(qs[i])
        for j in range(n):
            set_j = set(qs[j])
            inter = len(set_i & set_j)
            union = len(set_i | set_j)
            jac[i, j] = inter / union if union > 0 else 1.0
    return jac

def pairwise_label_dist(labels):
    labels = labels.view(-1, 1)
    diff = torch.abs(labels - labels.T) / (labels + labels.T + 1e-6)
    return diff

def train(estimator, element_emb, group_embs, degree, train_loader, val_loader, optimizer, regularizer, args, dev):

    best_loss = 1e50
    for epoch in range(args.qe):
        estimator.train()
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            qs, truth = batch
            truth = truth.float().to(dev)
            lst = [element_emb(q + 1) for q in qs]
            sel = [torch.log(degree[q] + 1e-6) for q in qs]
            sel = pad_sequence(sel, True).unsqueeze(-1).to(dev)
            lst = pad_sequence(lst, True).to(dev)

            pred, proj = estimator(lst, group_embs, sel)

            # 主损失
            l_train = weight_q_error(truth, pred).sum()
            l_contrastive = 0.0

            # -------- 对比学习部分 --------
            if args.qt != 'subset':
                jac_sim = pairwise_jaccard(qs)
                label_sim = pairwise_label_dist(truth)
                pos_mask = (jac_sim > args.pos_jac) | (label_sim < args.pos_label)
                diag = torch.eye(pos_mask.size(0), dtype=torch.bool, device=pos_mask.device)
                pos_mask[diag] = False
                proj = F.normalize(proj, dim=1)
                sim_matrix = torch.matmul(proj, proj.T) / args.temperature
                sim_matrix_exp = torch.exp(sim_matrix)
                sim_matrix_exp = sim_matrix_exp * (~diag)
                pos_sum = (sim_matrix_exp * pos_mask).sum(1)
                denom = sim_matrix_exp.sum(1) + 1e-8
                l_contrastive = -torch.log(pos_sum / (denom + 1e-8) + 1e-8).mean()
            # ----------------------------------

            l_reg = regularizer(estimator)
            train_loss = l_train + args.cl_weight * l_contrastive + l_reg
            train_loss.backward()
            optimizer.step()

        # Validation & Save best model
        if epoch > 9:
            estimator.eval()
            with torch.no_grad():
                val_preds = torch.tensor([]).to(dev)
                val_truth = torch.tensor([]).to(dev)
                for _, val_batch in enumerate(val_loader):
                    val_qs, val_cs = val_batch
                    val_truth = torch.cat((val_truth, val_cs.to(dev)))
                    val_lst = [element_emb(q + 1) for q in val_qs]
                    val_sel = [torch.log(degree[q] + 1e-6) for q in val_qs]
                    val_sel = pad_sequence(val_sel, True).unsqueeze(-1).to(dev)
                    val_lst = pad_sequence(val_lst, True).to(dev)
                    val_pred, _ = estimator(val_lst, group_embs, val_sel)
                    val_preds = torch.cat((val_preds, val_pred))
                val_loss = q_error_criterion(val_truth, val_preds).mean()
                print(epoch, val_loss.item(), val_preds.min().item(), val_preds.max().item())
                if val_loss.item() < best_loss and val_preds.min().item() > 0:
                    best_loss = val_loss.item()
                    workload_type = args.qt
                    dataset = args.d
                    distill_ratio = args.r
                    torch.save(estimator.state_dict(), f'save_model/{dataset}_estimator_{workload_type}_{distill_ratio}.pth')




def test(estimator, element_emb, group_embs, degree, test_loader, device, save_path="test_results.txt"):
    estimator.eval()
    preds, truth = [], []
    times = []
    result_lines = ["query,truth,predict,q-error,element_freq"]
    with torch.no_grad():
        for test_qs, test_cs in test_loader:
            start = time.perf_counter()
            truth.append(test_cs)

            test_lst = [element_emb(q + 1) for q in test_qs]
            test_sel = [torch.log(degree[q]) for q in test_qs]
            test_sel = pad_sequence(test_sel, True).unsqueeze(-1).to(device)
            test_lst = pad_sequence(test_lst, True).to(device)

            test_pred, *_ = estimator(test_lst, group_embs, test_sel)
            preds.append(test_pred.cpu())
            times.append((time.perf_counter() - start) * 1000)

            # 保存每条记录
            for i in range(len(test_qs)):
                q = test_qs[i].tolist()
                t = test_cs[i].item()
                p = test_pred[i].item()
                qerr = max(p / t, t / p) if t > 0 else 9999
                freqs = degree[test_qs[i]].cpu().tolist()
                result_lines.append(f"\"{q}\",{t:.6f},{p:.6f},{qerr:.6f},\"{freqs}\"")

    qerr = q_error_criterion(torch.cat(truth), torch.cat(preds))
    print("Test Results:")
    print("Mean: %.3f, Median: %.3f, 95%%: %.3f, 99%%: %.3f, Max: %.3f, Time(ms): %.3f" % (
        qerr.mean().item(), qerr.median().item(),
        qerr.quantile(0.95).item(), qerr.quantile(0.99).item(),
        qerr.max().item(), np.mean(times)
    ))

    # 保存文件
    with open(save_path, "w") as f:
        for line in result_lines:
            f.write(line + "\n")
    print(f"Saved test results to {save_path}")



def main():
    args = make_args()
    device = torch.device(f'cuda:{args.dev}' if args.dev >= 0 else 'cpu')
    seed_random(args.s)

    element_emb, group_embs = prepare_embeddings(args.d, args.dim, device)
    group_embs = group_embs[:int(0.7 * group_embs.size(0))]

    estimator = ACE(args.dim, args.cross_attn_num, args.self_attn_num, args.mlp_dep, args.mlp_dim).to(device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=args.ql)
    regularizer = ModelRegularizer(args.qr)
    degree = torch.from_numpy(np.load(f'{args.d}_freq.npy')).to(device)

    train_loader = load_workload(args.d, args.qt, args.qf, 'train', args.qb)
    val_loader = load_workload(args.d, args.qt, args.qf, 'val', args.qb)
    test_loader = load_workload(args.d, args.qt, args.qf, 'test', args.qb)
    workload_type = args.qt
    dataset = args.d
    distill_ratio = args.r
    if args.m == 'train':
        train(estimator, element_emb, group_embs, degree, train_loader, val_loader, optimizer, regularizer, args, device)
    else:
        estimator.load_state_dict(torch.load(f'save_model/%s_estimator_%s_%s.pth' % (dataset, workload_type, distill_ratio)))
        test(estimator, element_emb, group_embs, degree, test_loader, device)


if __name__ == '__main__':
    main()





def test(estimator, element_emb, group_embs, degree, test_loader, device, workload_type, save_path="test_results.txt"):
    estimator.eval()
    preds, truth = [], []
    times = []
    result_lines = ["query,truth,predict,q-error,element_freq"]

    # 计算度数的均值
    non_zero_degrees = degree[degree > 0].float()  # 转换为浮动类型
    degree_mean = non_zero_degrees.mean().item() if len(non_zero_degrees) > 0 else 1.0  # 如果没有非零度数，默认均值为1

    with torch.no_grad():
        for test_qs, test_cs in test_loader:
            start = time.perf_counter()
            truth.append(test_cs)

            # 获取测试元素的embedding
            test_lst = [element_emb(q + 1) for q in test_qs]
            
            # 替换查询中度数为0的元素的度数为均值
            test_sel = []
            for q in test_qs:
                # 使用torch.where来进行逐个元素的替换
                # 这里torch.where会在degree[q] > 0时选择degree[q]，否则选择degree_mean
                log_degree = torch.where(degree[q] > 0, torch.log(degree[q]), torch.log(torch.tensor(degree_mean, dtype=torch.float32)))
                test_sel.append(log_degree)
                
            test_sel = pad_sequence(test_sel, True).unsqueeze(-1).to(device)
            test_lst = pad_sequence(test_lst, True).to(device)

            # 进行预测
            test_pred, *_ = estimator(test_lst, group_embs, test_sel)
            preds.append(test_pred.cpu())
            times.append((time.perf_counter() - start) * 1000)

            # 保存每条记录
            for i in range(len(test_qs)):
                q = test_qs[i].tolist()
                t = test_cs[i].item()
                p = test_pred[i].item()
                qerr = max(p / t, t / p) if t > 0 else 9999
                freqs = degree[test_qs[i]].cpu().tolist()
                result_lines.append(f"\"{q}\",{t:.6f},{p:.6f},{qerr:.6f},\"{freqs}\"")

    qerr = q_error_criterion(torch.cat(truth), torch.cat(preds))
    print("Test Results:")
    print("Mean: %.3f, Median: %.3f, 95%%: %.3f, 99%%: %.3f, Max: %.3f, Time(ms): %.3f" % (
        qerr.mean().item(), qerr.median().item(),
        qerr.quantile(0.95).item(), qerr.quantile(0.99).item(),
        qerr.max().item(), np.mean(times)
    ))






    import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.preprocess import seed_random, read_data
from utils.criterion import weight_q_error, q_error_criterion
from utils.customized_dataset import MyQueryDataset
from utils.arg_parser import make_args
from model.ace import ACE
from model.encoder import ModelRegularizer
import matplotlib.pyplot as plt




# ====== 采样统计Jaccard分布 ======
def pairwise_jaccard(qs):
    n = len(qs)
    jac = torch.zeros((n, n), device='cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(n):
        set_i = set(qs[i])
        for j in range(n):
            set_j = set(qs[j])
            inter = len(set_i & set_j)
            union = len(set_i | set_j)
            jac[i, j] = inter / union if union > 0 else 1.0
    return jac

# ====== 采样统计label归一化差分布 ======
def pairwise_label_dist(labels):
    labels = labels.view(-1, 1)
    diff = torch.abs(labels - labels.T) / (labels + labels.T + 1e-6)
    return diff

def prepare_embeddings(dataset, dim, device):
    _, num_element = read_data(dataset)
    element_emb = nn.Embedding(num_element + 1, dim, padding_idx=0).requires_grad_(False)
    group_embs = torch.load(f'save_model/{dataset}_groups.pth').detach().to(device).requires_grad_(False)
    return element_emb, group_embs

def load_workload(dataset, workload_type, workload_freq, split, batch_size, nrows=None):
    if split in ['train', 'val']:
        path = f'./query/{dataset}/{workload_type}/{split}_workload.csv'
    else:
        path = f'./query/{dataset}/{workload_type}/{split}_{workload_freq}/{dataset}.csv'
    print(f"Loading from: {path}")
    data = pd.read_csv(path, header=None, nrows=nrows)
    dataset = MyQueryDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'), collate_fn=MyQueryDataset.collate_fn)

# ====== 自动搜索对比学习相关阈值 ======

def auto_select_jaccard_threshold(train_loader, sample_batches=20, percentiles=[0.1, 0.2, 0.2]):
    all_jaccards = []
    for idx, batch in enumerate(train_loader):
        if idx >= sample_batches:
            break
        qs, _ = batch
        jac = pairwise_jaccard(qs)
        arr = jac.cpu().numpy()
        arr = arr[~np.eye(arr.shape[0], dtype=bool)]
        all_jaccards.append(arr)
    all_jaccards = np.concatenate(all_jaccards)
    try:
        plt.hist(all_jaccards, bins=50)
        plt.title('Jaccard pairwise distribution')
        plt.xlabel('Jaccard')
        plt.show()
    except:
        pass
    rec_vals = [np.percentile(all_jaccards, p*100) for p in percentiles]
    print(f"自动建议的Jaccard阈值: {rec_vals}，分别对应{[int(p*100) for p in percentiles]}分位点")
    return rec_vals

def auto_select_label_threshold(train_loader, sample_batches=20, percentiles=[0.1, 0.2, 0.3]):
    all_diffs = []
    for idx, batch in enumerate(train_loader):
        if idx >= sample_batches:
            break
        _, truth = batch
        labels = truth.float()
        label_sim = pairwise_label_dist(labels)
        arr = label_sim.cpu().numpy()
        arr = arr[~np.eye(arr.shape[0], dtype=bool)]
        all_diffs.append(arr)
    all_diffs = np.concatenate(all_diffs)
    try:
        plt.hist(all_diffs, bins=50)
        plt.title('Label normalized diff distribution')
        plt.xlabel('label_sim')
        plt.show()
    except:
        pass
    rec_vals = [np.percentile(all_diffs, p*100) for p in percentiles]
    print(f"自动建议的label_sim阈值: {rec_vals}，分别对应{[int(p*100) for p in percentiles]}分位点")
    return rec_vals

def auto_select_freq_threshold(train_loader, element_emb, degree, sample_batches=20, percentiles=[0.1, 0.2, 0.3]):
    all_diffs = []
    for idx, batch in enumerate(train_loader):
        if idx >= sample_batches:
            break
        qs, truth = batch
        sel = [torch.log(degree[q] + 1e-6) for q in qs]
        sel = pad_sequence(sel, True).unsqueeze(-1)
        sel = sel.float()
        sel_mean = sel.mean(dim=1).squeeze(-1)  # [B]
        freq_diff = torch.abs(sel_mean.unsqueeze(1) - sel_mean.unsqueeze(0))  # [B, B]
        all_diffs.append(freq_diff.cpu().numpy().flatten())
    all_diffs = np.concatenate(all_diffs)
    all_diffs = all_diffs[all_diffs > 0]  # 去除对角线
    try:
        plt.hist(all_diffs, bins=50)
        plt.title('Query pair freq_mean diff distribution')
        plt.xlabel('freq_mean diff')
        plt.show()
    except:
        pass
    rec_vals = [np.percentile(all_diffs, p*100) for p in percentiles]
    print(f"自动建议的freq_diff阈值: {rec_vals}，分别对应{[int(p*100) for p in percentiles]}分位点")
    return rec_vals

# ====== 训练主流程 ======

def train(estimator, element_emb, group_embs, degree, train_loader, val_loader, optimizer, regularizer, args, dev,
          pos_jac, pos_label, freq_diff_threshold):
    best_loss = 1e50
    for epoch in range(args.qe):
        estimator.train()
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            qs, truth = batch
            truth = truth.float().to(dev)
            lst = [element_emb(q + 1) for q in qs]
            sel = [torch.log(degree[q] + 1e-6) for q in qs]
            sel = pad_sequence(sel, True).unsqueeze(-1).to(dev)
            lst = pad_sequence(lst, True).to(dev)

            pred, proj = estimator(lst, group_embs, sel)

            # 主损失
            l_train = weight_q_error(truth, pred).sum()

            # ----------- 对比学习 --------------
            jac_sim = pairwise_jaccard(qs)
            label_sim = pairwise_label_dist(truth)
            sel_mean = sel.mean(dim=1).squeeze(-1)  # [B]
            freq_diff = torch.abs(sel_mean.view(-1, 1) - sel_mean.view(1, -1))
            freq_sim_mask = freq_diff < freq_diff_threshold

            pos_mask = (jac_sim > pos_jac) | (label_sim < pos_label) | freq_sim_mask
            diag = torch.eye(pos_mask.size(0), dtype=torch.bool, device=pos_mask.device)
            pos_mask[diag] = False
            proj = F.normalize(proj, dim=1)
            sim_matrix = torch.matmul(proj, proj.T) / args.temperature
            sim_matrix_exp = torch.exp(sim_matrix)
            sim_matrix_exp = sim_matrix_exp * (~diag)
            pos_sum = (sim_matrix_exp * pos_mask).sum(1)
            denom = sim_matrix_exp.sum(1) + 1e-8
            l_contrastive = -torch.log(pos_sum / (denom + 1e-8) + 1e-8).mean()
            # ----------------------------------

            l_reg = regularizer(estimator)
            train_loss = l_train + args.cl_weight * l_contrastive + l_reg
            train_loss.backward()
            optimizer.step()

        # Validation & Save best model
        if epoch > 9:
            estimator.eval()
            with torch.no_grad():
                val_preds = torch.tensor([]).to(dev)
                val_truth = torch.tensor([]).to(dev)
                for _, val_batch in enumerate(val_loader):
                    val_qs, val_cs = val_batch
                    val_truth = torch.cat((val_truth, val_cs.to(dev)))
                    val_lst = [element_emb(q + 1) for q in val_qs]
                    val_sel = [torch.log(degree[q] + 1e-6) for q in val_qs]
                    val_sel = pad_sequence(val_sel, True).unsqueeze(-1).to(dev)
                    val_lst = pad_sequence(val_lst, True).to(dev)
                    val_pred, _ = estimator(val_lst, group_embs, val_sel)
                    val_preds = torch.cat((val_preds, val_pred))
                val_loss = q_error_criterion(val_truth, val_preds).mean()
                print(epoch, val_loss.item(), val_preds.min().item(), val_preds.max().item())
                if val_loss.item() < best_loss and val_preds.min().item() > 0:
                    best_loss = val_loss.item()
                    workload_type = args.qt
                    dataset = args.d
                    distill_ratio = args.r
                    torch.save(estimator.state_dict(), f'save_model/{dataset}_estimator_{workload_type}_{distill_ratio}.pth')


def test(estimator, element_emb, group_embs, degree, test_loader, device, workload_type, save_path="test_results.txt"):
    estimator.eval()
    preds, truth = [], []
    times = []
    result_lines = ["query,truth,predict,q-error,element_freq"]

    # 计算度数的均值
    non_zero_degrees = degree[degree > 0].float()  # 转换为浮动类型
    degree_mean = non_zero_degrees.mean().item() if len(non_zero_degrees) > 0 else 1.0  # 如果没有非零度数，默认均值为1

    with torch.no_grad():
        for test_qs, test_cs in test_loader:
            start = time.perf_counter()
            truth.append(test_cs)

            # 获取测试元素的embedding
            test_lst = [element_emb(q + 1) for q in test_qs]
            
            # 替换查询中度数为0的元素的度数为均值
            test_sel = []
            for q in test_qs:
                # 使用torch.where来进行逐个元素的替换
                # 这里torch.where会在degree[q] > 0时选择degree[q]，否则选择degree_mean
                log_degree = torch.where(degree[q] > 0, torch.log(degree[q]), torch.log(torch.tensor(degree_mean, dtype=torch.float32)))
                test_sel.append(log_degree)
                
            test_sel = pad_sequence(test_sel, True).unsqueeze(-1).to(device)
            test_lst = pad_sequence(test_lst, True).to(device)

            # 进行预测
            test_pred, *_ = estimator(test_lst, group_embs, test_sel)
            preds.append(test_pred.cpu())
            times.append((time.perf_counter() - start) * 1000)

            # 保存每条记录
            for i in range(len(test_qs)):
                q = test_qs[i].tolist()
                t = test_cs[i].item()
                p = test_pred[i].item()
                qerr = max(p / t, t / p) if t > 0 else 9999
                freqs = degree[test_qs[i]].cpu().tolist()
                result_lines.append(f"\"{q}\",{t:.6f},{p:.6f},{qerr:.6f},\"{freqs}\"")

    qerr = q_error_criterion(torch.cat(truth), torch.cat(preds))
    print("Test Results:")
    print("Mean: %.3f, Median: %.3f, 95%%: %.3f, 99%%: %.3f, Max: %.3f, Time(ms): %.3f" % (
        qerr.mean().item(), qerr.median().item(),
        qerr.quantile(0.95).item(), qerr.quantile(0.99).item(),
        qerr.max().item(), np.mean(times)
    ))

    # 如果需要，将结果保存到文件
    with open(save_path, 'w') as f:
        for line in result_lines:
            f.write(line + "\n")




def main():
    args = make_args()
    device = torch.device(f'cuda:{args.dev}' if args.dev >= 0 else 'cpu')
    seed_random(args.s)

    element_emb, group_embs = prepare_embeddings(args.d, args.dim, device)
    group_embs = group_embs[:int(0.7 * group_embs.size(0))]

    estimator = ACE(args.dim, args.cross_attn_num, args.self_attn_num, args.mlp_dep, args.mlp_dim).to(device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=args.ql)
    regularizer = ModelRegularizer(args.qr)
    degree = torch.from_numpy(np.load(f'{args.d}_freq.npy')).to(device)

    train_loader = load_workload(args.d, args.qt, args.qf, 'train', args.qb)
    val_loader = load_workload(args.d, args.qt, args.qf, 'val', args.qb)
    test_loader = load_workload(args.d, args.qt, args.qf, 'test', args.qb)

    print("=== 自动搜索所有对比学习阈值 ===")
    jaccard_candidates = auto_select_jaccard_threshold(train_loader)
    label_sim_candidates = auto_select_label_threshold(train_loader)
    freq_diff_candidates = auto_select_freq_threshold(train_loader, element_emb, degree)
    # 这里默认用中间值(比如20%/80%分位点)，可以手动选择
    pos_jac = jaccard_candidates[1]
    pos_label = label_sim_candidates[1]
    freq_diff_threshold = freq_diff_candidates[1]
    print(f"训练用Jaccard阈值: {pos_jac:.4f} label阈值: {pos_label:.4f} freq阈值: {freq_diff_threshold:.4f}")

    workload_type = args.qt
    dataset = args.d
    distill_ratio = args.r
    if args.m == 'train':
        train(estimator, element_emb, group_embs, degree, train_loader, val_loader, optimizer, regularizer, args, device,
              pos_jac, pos_label, freq_diff_threshold)
    else:
        estimator.load_state_dict(torch.load(f'save_model/%s_estimator_%s_%s.pth' % (dataset, workload_type, distill_ratio)))
        test(estimator, element_emb, group_embs, degree, test_loader, device,workload_type)

if __name__ == '__main__':
    main()



import torch
import torch.nn as nn
import torch.nn.functional as F
from model.att_modules import *
from model.encoder import *

class ACE(nn.Module):
    def __init__(self, dim, cross_att_layers, self_att_layers, mlp_layers, mlp_dim, data_num=None, agg_model=None):
        super().__init__()
        self.analyzer1 = nn.ModuleList([
            nn.ModuleList([PostNormAttention(dim, dim, hidden_dim=dim), PostNormFFN(dim)])
            for _ in range(cross_att_layers)
        ])
        self.analyzer2 = nn.ModuleList([
            nn.ModuleList([PostNormAttention(dim, dim, hidden_dim=dim), PostNormFFN(dim)])
            for _ in range(self_att_layers)
        ])

        self.pool = Pooling(dim + 1, 1)
        # 新增：统计特征拼接，维度+4
        mlp_input_dim = dim + 4+1  # pooling输出 dim + 4(统计特征)
        self.mlp = BasicMLP([mlp_input_dim] + [mlp_dim] * (mlp_layers - 1) + [1])
        self.projection_head = ProjectionHead(mlp_input_dim)

    def forward(self, q, x, sel):
        q = torch.clamp(q, -50, 50)
        x = torch.clamp(x, -50, 50)
        sel = torch.clamp(sel, -50, 50)

        x = x.repeat(q.size(0), 1, 1)
        q_mask = torch.sign(torch.sum(torch.abs(q), dim=-1))  # [B, L]

        for attn, ffn in self.analyzer1:
            q = torch.nan_to_num(q, nan=0.0, posinf=1e3, neginf=-1e3)
            q = attn(q, x)
            q = ffn(q)

        for attn, ffn in self.analyzer2:
            q = torch.nan_to_num(q, nan=0.0, posinf=1e3, neginf=-1e3)
            q = attn(q)
            q = ffn(q)

        mask = q_mask.unsqueeze(-1).expand(-1, -1, q.size(-1))
        q = q * mask
        q = torch.cat([q, sel], dim=-1)
        q = self.pool(q, q_mask)  # [B, D]

        # --- 拼接统计特征 ---
        lengths = q_mask.sum(dim=1, keepdim=True)              # [B, 1]
        log_lengths = torch.log(lengths + 1e-6)                # [B, 1]
        sel_mean = sel.sum(dim=1) / (lengths + 1e-6)           # [B, 1]
        sel_max = sel.max(dim=1)[0]                            # [B, 1]
        sel_min = sel.min(dim=1)[0]                            # [B, 1]
        q_feat = torch.cat([q, log_lengths, sel_mean, sel_max, sel_min], dim=1)  # [B, D+4]
        q_feat = torch.nan_to_num(q_feat, nan=0.0, posinf=1e3, neginf=-1e3)
        res = self.mlp(q_feat)
        proj = self.projection_head(q_feat)
        return res, proj



class Pooling(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.mab = Attention(dim, dim, dim, heads)
        self.pooling_emb = nn.Parameter(torch.zeros(1, dim))
        self.pool = nn.AdaptiveAvgPool2d((1, dim))
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.ffn = BasicMLP([dim, 512, dim])
        nn.init.trunc_normal_(self.pooling_emb, std=0.02)

    def forward(self, x, mask=None):
        x = torch.clamp(x, -50, 50)
        out = self.pool(x)
        out = self.norm1(out + self.mab(out, x, k_mask=mask))
        out = self.norm2(out + self.ffn(out))
        return torch.nan_to_num(out.squeeze(1), nan=0.0, posinf=1e3, neginf=-1e3)



