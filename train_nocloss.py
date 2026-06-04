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
    group_model = torch.load(f'save_model/{dataset}_groups.pth')
    if isinstance(group_model, torch.nn.Embedding):
        group_embs = group_model.weight.detach().to(device).requires_grad_(False)
    else:
        group_embs = group_model.detach().to(device).requires_grad_(False)

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
    # print(f"自动建议的Jaccard阈值: {rec_vals}，分别对应{[int(p*100) for p in percentiles]}分位点")
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
    # print(f"自动建议的label_sim阈值: {rec_vals}，分别对应{[int(p*100) for p in percentiles]}分位点")
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
    # print(f"自动建议的freq_diff阈值: {rec_vals}，分别对应{[int(p*100) for p in percentiles]}分位点")
    return rec_vals

# ====== 训练主流程 ======

def train(estimator, element_emb, group_embs, degree, train_loader, val_loader, optimizer, regularizer, args, dev,
          pos_jac, pos_label, freq_diff_threshold):
    best_loss = 1e50
    import time
    start_time = time.time()

    # ✅ 在第一个 epoch 中收集全部投影向量
    all_proj, all_truth = [], []

    for epoch in range(args.qe):
        estimator.train()
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            qs, truth = batch
            truth = truth.float().to(dev)
            lst = [element_emb(q + 1) for q in qs]
            sel = [torch.log(degree[q] + 1e-6) for q in qs]
            sel = pad_sequence(sel, True).unsqueeze(-1).to(dev)
            lst = pad_sequence(lst, True).to(dev)

            pred, proj = estimator(lst, group_embs, sel)

            # ✅ 收集整个训练集 proj 和 truth（仅在第一个 epoch）
            if epoch == 0:
                all_proj.append(proj.detach().cpu())
                all_truth.append(truth.detach().cpu())

            # 主损失
            l_train = weight_q_error(truth, pred).sum()

            # 对比学习损失
            jac_sim = pairwise_jaccard(qs)
            label_sim = pairwise_label_dist(truth)
            sel_mean = sel.mean(dim=1).squeeze(-1)
            freq_diff = torch.abs(sel_mean.view(-1, 1) - sel_mean.view(1, -1))
            freq_sim_mask = freq_diff < freq_diff_threshold

            pos_mask = (jac_sim > pos_jac) | (label_sim < pos_label) | freq_sim_mask
            diag = torch.eye(pos_mask.size(0), dtype=torch.bool, device=pos_mask.device)
            pos_mask[diag] = False
            proj = F.normalize(proj, dim=1)
            sim_matrix = torch.matmul(proj, proj.T) / args.temperature
            sim_matrix_exp = torch.exp(sim_matrix) * (~diag)
            pos_sum = (sim_matrix_exp * pos_mask).sum(1)
            denom = sim_matrix_exp.sum(1) + 1e-8
            l_contrastive = -torch.log(pos_sum / (denom + 1e-8) + 1e-8).mean()

            l_reg = regularizer(estimator)
            train_loss = l_train + args.cl_weight * l_contrastive + l_reg
            train_loss.backward()
            optimizer.step()

        # ✅ 保存 proj 向量用于 t-SNE 可视化（在第一个 epoch 后保存）
        if epoch == 0 and len(all_proj) > 0:
            proj_tensor = torch.cat(all_proj, dim=0)
            truth_tensor = torch.cat(all_truth, dim=0)
            torch.save({
                'proj': proj_tensor,
                'labels': truth_tensor
            }, f'proj_with_cl.pt' if args.cl_weight > 0 else f'proj_no_cl.pt')
            print(f"✅ 已保存训练集全部 proj（{proj_tensor.shape[0]} 条样本）")

        # 验证集评估
        if epoch > 9:
            estimator.eval()
            with torch.no_grad():
                val_preds, val_truth = torch.tensor([]).to(dev), torch.tensor([]).to(dev)
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
                    torch.save(estimator.state_dict(),
                               f'save_model/{args.d}_estimator_{args.qt}_{args.r}.pth')

    print(f"\n✅ Total training time: {time.time() - start_time:.2f} seconds")




def test(estimator, element_emb, group_embs, degree, test_loader, device, workload_type, save_path="test_results.txt"):
    estimator.eval()
    preds, truth = [], []
    times = []
    result_lines = ["query,truth,predict,q-error,element_freq,inference_time(ms)"]

    # 计算度数的均值
    non_zero_degrees = degree[degree > 0].float()
    degree_mean = non_zero_degrees.mean().item() if len(non_zero_degrees) > 0 else 1.0

    with torch.no_grad():
        for test_qs, test_cs in test_loader:
            for i in range(len(test_qs)):
                q = test_qs[i]
                t = test_cs[i].item()
                freqs = degree[q].cpu().tolist()

                # 开始计时
                start = time.perf_counter()

                q_emb = element_emb(q + 1).unsqueeze(0).to(device)  # [1, seq_len, dim]
                q_deg = torch.where(degree[q] > 0, torch.log(degree[q]),
                                    torch.log(torch.tensor(degree_mean, dtype=torch.float32)))
                q_deg = q_deg.unsqueeze(-1).unsqueeze(0).to(device)  # [1, seq_len, 1]
                pred, *_ = estimator(q_emb, group_embs, q_deg)

                # 结束计时
                elapsed = (time.perf_counter() - start) * 1000  # 单位 ms
                times.append(elapsed)

                p = pred.item()
                qerr = max(p / t, t / p) if t > 0 else 9999

                # 保存每条记录（含推理时间）
                result_lines.append(f"\"{q.tolist()}\",{t:.6f},{p:.6f},{qerr:.6f},\"{freqs}\",{elapsed:.3f}")

                # 同时收集预测和真实值
                preds.append(torch.tensor([p]))
                truth.append(torch.tensor([t]))

    qerr = q_error_criterion(torch.cat(truth), torch.cat(preds))
    print("Test Results:")
    print("Mean: %.3f, Median: %.3f, 95%%: %.3f, 99%%: %.3f, Max: %.3f, Time(ms): %.3f" % (
        qerr.mean().item(), qerr.median().item(),
        qerr.quantile(0.95).item(), qerr.quantile(0.99).item(),
        qerr.max().item(), np.mean(times)
    ))

    # 保存每条查询记录（含推理时间）到文件
    with open(save_path, 'w') as f:
        for line in result_lines:
            f.write(line + "\n")

    # 打印总推理时间与平均时间
    total_time_ms = np.sum(times)
    avg_time_ms = np.mean(times)
    print(f"\n⏱️ 推理总耗时: {total_time_ms:.2f} ms")
    print(f"🚀 平均每条查询耗时: {avg_time_ms:.2f} ms")


# def test(estimator, element_emb, group_embs, degree, test_loader, device, workload_type, save_path="test_results.txt"):
#     estimator.eval()
#     preds, truth = [], []
#     times = []
#     result_lines = ["query,truth,predict,q-error,element_freq"]

#     # 计算度数的均值
#     non_zero_degrees = degree[degree > 0].float()  # 转换为浮动类型
#     degree_mean = non_zero_degrees.mean().item() if len(non_zero_degrees) > 0 else 1.0  # 如果没有非零度数，默认均值为1

#     with torch.no_grad():
#         for test_qs, test_cs in test_loader:
#             start = time.perf_counter()
#             truth.append(test_cs)

#             # 获取测试元素的embedding
#             test_lst = [element_emb(q + 1) for q in test_qs]
            
#             # **不再修改度数**，直接使用度数
#             test_sel = []
#             for q in test_qs:
#                 # 直接使用度数，无需替换
#                 log_degree = torch.log(degree[q].float())  # 将度数转换为浮动类型并计算log
#                 test_sel.append(log_degree)
                
#             test_sel = pad_sequence(test_sel, True).unsqueeze(-1).to(device)
#             test_lst = pad_sequence(test_lst, True).to(device)

#             # 进行预测
#             test_pred, *_ = estimator(test_lst, group_embs, test_sel)
#             preds.append(test_pred.cpu())
#             times.append((time.perf_counter() - start) * 1000)

#             # 保存每条记录
#             for i in range(len(test_qs)):
#                 q = test_qs[i].tolist()
#                 t = test_cs[i].item()
#                 p = test_pred[i].item()
#                 qerr = max(p / t, t / p) if t > 0 else 9999
#                 freqs = degree[test_qs[i]].cpu().tolist()
#                 result_lines.append(f"\"{q}\",{t:.6f},{p:.6f},{qerr:.6f},\"{freqs}\"")

#     qerr = q_error_criterion(torch.cat(truth), torch.cat(preds))
#     print("Test Results:")
#     print("Mean: %.3f, Median: %.3f, 95%%: %.3f, 99%%: %.3f, Max: %.3f, Time(ms): %.3f" % (
#         qerr.mean().item(), qerr.median().item(),
#         qerr.quantile(0.95).item(), qerr.quantile(0.99).item(),
#         qerr.max().item(), np.mean(times)
#     ))

#     # 如果需要，将结果保存到文件
#     with open(save_path, 'w') as f:
#         for line in result_lines:
#             f.write(line + "\n")

def main():
    args = make_args()
    device = torch.device(f'cuda:{args.dev}' if args.dev >= 0 else 'cpu')
    seed_random(args.s)
    element_emb, group_embs = prepare_embeddings(args.d, args.dim, device)
    print("element_emb size:", element_emb.num_embeddings)

    group_embs = group_embs[:int(0.7 * group_embs.size(0))]

    estimator = ACE(args.dim, args.cross_attn_num, args.self_attn_num, args.mlp_dep, args.mlp_dim).to(device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=args.ql)
    regularizer = ModelRegularizer(args.qr)
    degree = torch.from_numpy(np.load(f'{args.d}_freq.npy')).to(device)

    train_loader = load_workload(args.d, args.qt, args.qf, 'train', args.qb)
    val_loader = load_workload(args.d, args.qt, args.qf, 'val', args.qb)
    test_loader = load_workload(args.d, args.qt, args.qf, 'test', args.qb)
    
    # print("=== 自动搜索所有对比学习阈值 ===")
    jaccard_candidates = auto_select_jaccard_threshold(train_loader)
    label_sim_candidates = auto_select_label_threshold(train_loader)
    freq_diff_candidates = auto_select_freq_threshold(train_loader, element_emb, degree)
    # 这里默认用中间值(比如20%/80%分位点)，可以手动选择
    pos_jac = jaccard_candidates[1]
    pos_label = label_sim_candidates[1]
    freq_diff_threshold = freq_diff_candidates[1]
    # print(f"训练用Jaccard阈值: {pos_jac:.4f} label阈值: {pos_label:.4f} freq阈值: {freq_diff_threshold:.4f}")

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
