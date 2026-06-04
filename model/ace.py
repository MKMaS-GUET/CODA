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
