from model.att_modules import *
from model.modules import *


# class Aggregator(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         # self.ffn = BasicMLP([dim] + [512] * (depth - 1) + [dim])
#         self.ffn = nn.Linear(dim, dim)
#         # mean pooling
#         self.pool = nn.AdaptiveAvgPool2d((1, dim))
#         # # max pooling
#         # self.pool = nn.AdaptiveMaxPool2d((1, dim))

    
#     def forward(self, x, m=None):
#         x = self.ffn(x)
#         if m is not None:
#             x = x * m
#         x = F.gelu(x)
        
#         # used in max pooling
#         # max_neg_value = torch.finfo(x.dtype).min
#         # x[x == 0] = max_neg_value
        
#         x = self.pool(x).squeeze(1)
#         x = F.normalize(x)
        
#         return x


class Aggregator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.ffn = BasicMLP([dim] + [512] * (depth - 1) + [dim])
        self.ffn = nn.Linear(dim, dim)
        self.pool = nn.AdaptiveAvgPool2d((1, dim))
        self.projection_head = ProjectionHead(dim)  # 添加投影头，用于对比学习

    def forward(self, x, m=None):
        x = self.ffn(x)
        if m is not None:
            x = x * m
        x = F.gelu(x)
        x = self.pool(x).squeeze(1)
        x = F.normalize(x)
        proj = self.projection_head(x)  # 得到对比表示向量
        return x, proj  # 返回原始特征与对比向量


class PostDistillation(nn.Module):
    def __init__(self, depth, latent_dim, data_dim, first_share=False):
        super().__init__()
        self.cross_attn_1 = PostNormAttention(latent_dim, data_dim)
        self.cross_ffn_1 = PostNormFFN(latent_dim)
        
        if not first_share:
            self.cross_attn_n = PostNormAttention(latent_dim, data_dim)
            self.cross_ffn_n = PostNormFFN(latent_dim)

        self.weight_share = first_share
        self.depth = depth
    
    
    def forward(self, data, latents):
        latents = latents.repeat(data.size(0), 1, 1)
        latents = self.cross_attn_1(latents, data)
        latents = self.cross_ffn_1(latents)

        cross_attn_n = self.cross_attn_1 if self.weight_share else self.cross_attn_n
        cross_ffn_n = self.cross_ffn_1 if self.weight_share else self.cross_ffn_n

        for _ in range(1, self.depth):
            latents = cross_attn_n(latents, data)
            latents = cross_ffn_n(latents)

        return latents.squeeze(0)
    

class Featurization(nn.Module):
    def __init__(self, distill_depth, data_dim, latent_dim, distill_ratio, distill_share=False):
        super().__init__()
        self.distill_ratio = distill_ratio
        self.aggregator = Aggregator(data_dim)
        # self.distillation = Perceiver(distill_depth, latent_dim, data_dim, distill_share)
        self.distillation = PostDistillation(distill_depth, latent_dim, data_dim, distill_share)
    

    def forward(self, x, pos_samples, neg_samples, mask, distilled_embs=None):
        set_embs = self.aggregator(x, mask)

        concat_samples = torch.cat((pos_samples, neg_samples), dim=1)
        link_pred = (set_embs.unsqueeze(1) * concat_samples).sum(-1)
        
        if distilled_embs is None:
            distilled_embs = torch.randperm(set_embs.size(0))[:max(int(set_embs.size(0) * self.distill_ratio), 2)]
            # distilled_embs = torch.randperm(set_embs.size(0))[:min(set_embs.size(0), self.distill_size)]
            distilled_embs = set_embs[distilled_embs].detach()
        
        distilled_embs = self.distillation(set_embs.unsqueeze(0), distilled_embs)
        
        return link_pred, set_embs, distilled_embs
    
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim)
        )

    def forward(self, x):
        return self.projection(x)
    
class ModelRegularizer(nn.Module):
    def __init__(self, reg_weight):
        super().__init__()
        self.reg_weight = reg_weight

    def forward(self, model):
        reg_loss = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                reg_loss += torch.norm(param, p=2)
        return self.reg_weight * reg_loss


