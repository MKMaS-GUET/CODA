import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image

def load_proj(file_path):
    data = torch.load(file_path)
    return data['proj'].numpy(), data['labels'].numpy()

# 加载数据
proj_with, label_with = load_proj("proj_with_cl.pt")
proj_no, label_no = load_proj("proj_no_cl.pt")

# t-SNE 2D 降维
X = np.vstack([proj_with, proj_no])
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='random', learning_rate='auto')
X_2d = tsne.fit_transform(X)

# 拆分
n = len(proj_with)
X_with = X_2d[:n]
X_no = X_2d[n:]

# 标签做 log1p 缩放
label_with_log = np.log1p(label_with)
label_no_log = np.log1p(label_no)

# 设置统一色阶范围
vmin = min(label_with_log.min(), label_no_log.min())
vmax = max(label_with_log.max(), label_no_log.max())

# 保存路径
file_with = "/workspace/ACE /imgage/tsne_2d_with_contrastive.png"
file_no = "/workspace/ACE /imgage/tsne_2d_no_contrastive.png"
file_combined = "/workspace/ACE /imgage/tsne_2d_combined.png"

# 设置配色
cmap_with = 'Spectral'
cmap_no = 'spring'

# 有对比学习图（2D）
plt.figure(figsize=(6, 5))
sc = plt.scatter(X_with[:, 0], X_with[:, 1],
                 c=label_with_log, cmap=cmap_with,
                 s=20, alpha=0.8, vmin=vmin, vmax=vmax)
plt.title("With Contrastive Learning")
plt.colorbar(sc, shrink=0.8, label="Log(Label + 1)")
plt.tight_layout()
plt.savefig(file_with, dpi=300)
plt.close()

# 无对比学习图（2D）
plt.figure(figsize=(6, 5))
sc = plt.scatter(X_no[:, 0], X_no[:, 1],
                 c=label_no_log, cmap=cmap_no,
                 s=20, alpha=0.8, vmin=vmin, vmax=vmax)
plt.title("Without Contrastive Learning")
plt.colorbar(sc, shrink=0.8, label="Log(Label + 1)")
plt.tight_layout()
plt.savefig(file_no, dpi=300)
plt.close()

# 拼接图像
img1 = Image.open(file_with)
img2 = Image.open(file_no)
h = max(img1.height, img2.height)
combined = Image.new('RGB', (img1.width + img2.width, h))
combined.paste(img1, (0, 0))
combined.paste(img2, (img1.width, 0))
combined.save(file_combined)

print(f"✅ 拼接图已保存为：{file_combined}")
