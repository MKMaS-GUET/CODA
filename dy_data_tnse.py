import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

from dy_query_analyzer import AlignNet
from utils.arg_parser import make_args

def visualize_embedding_3d(dataset, workload_type, dim, device):
    # 加载 base 和 new group embedding
    base_path = f'save_model/{dataset}_groups.pth'
    new_path = f'save_model/{dataset}_groups_new.pth'
    base_group_raw = torch.load(base_path, map_location=device)
    new_group = torch.load(new_path, map_location=device)

    print(f"📦 原始 Base Group 大小：{len(base_group_raw)}")
    print(f"📦 原始 New Group 大小：{len(new_group)}")

    # 使用全部 new_group，base_group 只截取同长度用于对比可视化（不影响原始数据）
    base_group = base_group_raw[:len(new_group)]
    print(f"📌 实际用于 t-SNE 的点数：Base = {len(base_group)}, New = {len(new_group)}")

    # 加载训练好的 AlignNet 参数
    align_net = AlignNet(dim).to(device)
    align_path = f'save_model/{dataset}_{workload_type}_align_net.pth'
    align_net.load_state_dict(torch.load(align_path, map_location=device))
    align_net.eval()

    # 对齐后的新 group
    with torch.no_grad():
        new_group_aligned = align_net(new_group)

    # 拼接进行 t-SNE 三维降维
    all_embeddings = torch.cat([base_group, new_group, new_group_aligned], dim=0).cpu().numpy()
    print("🔍 正在进行 t-SNE 三维降维...")
    tsne = TSNE(n_components=3, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(all_embeddings)

    n = len(new_group)
    base_3d = tsne_result[:n]
    new_raw_3d = tsne_result[n:2*n]
    new_aligned_3d = tsne_result[2*n:]

    # 可视化
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=140)
    ax.scatter(base_3d[:, 0], base_3d[:, 1], base_3d[:, 2], c='blue', label='Base Group', s=40, alpha=0.8)
    ax.scatter(new_raw_3d[:, 0], new_raw_3d[:, 1], new_raw_3d[:, 2], c='red', label='New Group (No Align)', s=40, alpha=0.8)
    ax.scatter(new_aligned_3d[:, 0], new_aligned_3d[:, 1], new_aligned_3d[:, 2], c='green', label='New Group (Aligned)', s=40, alpha=0.8)

   
    ax.legend()
    plt.tight_layout()

    # 保存图像
    save_dir = f'vis_output/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/embedding_alignment_tsne3d_{workload_type}.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 3D 可视化图已保存到 {save_path}")

if __name__ == '__main__':
    args = make_args()
    dataset = args.d
    workload_type = args.qt
    dim = args.dim
    device = torch.device(f'cuda:{args.dev}' if args.dev >= 0 and torch.cuda.is_available() else 'cpu')

    visualize_embedding_3d(dataset, workload_type, dim, device)
