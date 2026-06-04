import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FullyConnectNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FullyConnectNN, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.layer.weight)
        nn.init.constant_(self.layer.bias, 0.0)

    def forward(self, input):
        return self.layer(input)


# class BasicMLP(nn.Module):
#     def __init__(self, dims, act='relu', dropout=0.):
#         super(BasicMLP, self).__init__()
#         self.layers = nn.ModuleList()
#         for i in range(1, len(dims)):
#             self.layers.append(FullyConnectNN(dims[i - 1], dims[i]))
#         self.act = getattr(F, act)
#         if dropout:
#             self.dropout = nn.Dropout(dropout)
#         else:
#             self.dropout = None

#     def forward(self, input):
#         curr_input = input
#         for i in range(len(self.layers) - 1):
#             hidden = self.layers[i](curr_input)
#             hidden = self.act(hidden)
#             if self.dropout:
#                 hidden = self.dropout(hidden)
#             curr_input = hidden
#         output = self.layers[-1](curr_input)
#         return output
import torch
import torch.nn as nn
import torch.nn.functional as F

class FullyConnectNN(nn.Module):
    def __init__(self, in_features, out_features):
        super(FullyConnectNN, self).__init__()
        self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.layer(x)

class BasicMLP(nn.Module):
    def __init__(self, dims, act='relu', dropout=0.):
        super(BasicMLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(dims)):
            self.layers.append(FullyConnectNN(dims[i - 1], dims[i]))
        self.act = getattr(F, act)
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.input_dim = dims[0]

    def forward(self, input):
        # 自动断言特征数对齐，debug更方便
        assert input.shape[-1] == self.input_dim, \
            f"Input dim mismatch! Got {input.shape[-1]}, expected {self.input_dim}"
        curr_input = input
        for i in range(len(self.layers) - 1):
            hidden = self.layers[i](curr_input)
            hidden = self.act(hidden)
            if self.dropout:
                hidden = self.dropout(hidden)
            curr_input = hidden
        output = self.layers[-1](curr_input)
        return output


class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads):
        super(Attention, self).__init__()
        self.dim_v = value_dim
        self.num_heads = num_heads
        
        self.nn_q = nn.Linear(query_dim, value_dim)
        self.nn_k = nn.Linear(key_dim, value_dim)
        self.nn_v = nn.Linear(key_dim, value_dim)
        # self.nn_o = nn.Linear(value_dim, value_dim)
        self.nn_o = nn.Linear(value_dim, query_dim)
    
    def forward(self, q, x, q_mask=None, k_mask=None):
        query = self.nn_q(q)
        key = self.nn_k(x)
        value = self.nn_v(x)

        # split the dim (multi-head): equivalent to the original workflow (calculate n different result)
        query = self.split(query, self.num_heads)
        key = self.split(key, self.num_heads)
        value = self.split(value, self.num_heads)
        
        if q_mask is not None:
            q_mask = q_mask.unsqueeze(-1)
            q_mask = q_mask.expand(-1, -1, x.size(1))
            q_mask = q_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        if k_mask is not None:
            k_mask = k_mask.unsqueeze(1).expand(-1, q.size(1), -1)
            k_mask = k_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # scale dot product attention
        out = self.product_att(query, key, value, q_mask, k_mask)

        # concat and use the linear layer
        out = self.multi_head_concat(out)
        out = self.nn_o(out)

        return out

    def split(self, t, num_heads):
        batch_size, length, dim_value = t.size()
        
        dim_per_head = dim_value // num_heads
        
        assert ((dim_per_head) * num_heads == dim_value), 'Embedding size needs to be divisible by num_heads'
        t = t.view(batch_size, length, num_heads, dim_per_head).transpose(1, 2)
        return t

    def product_att(self, q, k, v, q_mask, k_mask):
        attn = (q @ k.transpose(2, 3)) / math.sqrt(k.size(-1))
        if k_mask is not None:
            attn = attn.masked_fill(k_mask==0., float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        # query mask
        if q_mask is not None:
            attn = attn * q_mask
        v = attn @ v
            
        return v
    
    def multi_head_concat(self, t):
        batch_size, _, length, _ = t.size()
        t = t.transpose(1, 2).contiguous().view(batch_size, length, self.dim_v)
        return t


class Block(nn.Module):
    def __init__(self, nl, qd, kd, vd, nh):
        super(Block, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(nl):
            self.layers.append(nn.ModuleList([Attention(qd, kd, vd, nh), nn.LayerNorm(kd), BasicMLP([kd, 512, kd], 'gelu'), nn.LayerNorm(kd)]))

        # self.num_layers = nl
        # self.attn = Attention(qd, kd, vd, nh)
        # self.ffn = BasicMLP([vd, 512, vd], 'gelu')
        # self.norm1 = nn.LayerNorm(vd)
        # self.norm2 = nn.LayerNorm(vd)
    
    def forward(self, q, x):
        for _, (attn, norm1, ffn, norm2) in enumerate(self.layers):
            q = norm1(q + attn(q, x))
            q = norm2(q + ffn(q))
            
        # for _ in range(self.num_layers):
        #     att_out = self.attn(q, x)
        #     q = self.norm1(q + att_out)
        #     q = self.norm2(q + self.ffn(q))
        
        return q


class AttentionPooling(nn.Module):
    def __init__(self, dim, heads=4):
        super(AttentionPooling, self).__init__()
        self.mab = Attention(dim, dim, dim, heads)
        # self.pooling_emb = nn.Parameter(torch.zeros(1, dim))
        # nn.init.trunc_normal_(self.pooling_emb, std=0.02)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # self.norm3 = nn.LayerNorm(dim)
        self.ffn = BasicMLP([dim, 512, dim])
        self.pool = nn.AdaptiveAvgPool2d((1, dim))

    # def forward(self, x, out, mask=None):
    #     # out = self.pooling_emb.repeat(x.size(0), 1, 1)
    #     # att_out = self.mab(out, self.norm1(x))
    #     # out = self.norm2(self.pool(x) + att_out)
    #     out = self.norm1(out + self.mab(out, x, k_mask=mask))
    #     out = self.norm2(out + self.ffn(out))
    #     # out = self.norm3(out)

    #     return out

    def forward(self, x, mask=None):
        # out = self.pooling_emb.repeat(x.size(0), 1, 1)
        # att_out = self.mab(out, self.norm1(x))
        # out = self.norm2(self.pool(x) + att_out)
        out = self.pool(x)
        out = self.norm1(self.pool(x) + self.mab(out, x, k_mask=mask))
        out = self.norm2(out + self.ffn(out))
        # out = self.norm3(out)

        return out


class PreNormAttention(nn.Module):
    def __init__(self, dim, data_dim=None, hidden_dim=512, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if data_dim is not None:
            self.norm_data = nn.LayerNorm(data_dim)
            self.attention = Attention(dim, data_dim, hidden_dim, num_heads)
        else:
            self.attention = Attention(dim, dim, hidden_dim, num_heads)
        # self.out = nn.Linear(hidden_dim, dim)


    def forward(self, x, d=None, mask=None):
        x = self.norm(x)
        if d is not None:
            d = self.norm_data(d)
            att_out = self.attention(x, d, mask)
        else:
            att_out = self.attention(x, x, mask)
        # att_out = self.out(att_out)
        
        return att_out
    

class PostNormAttention(nn.Module):
    def __init__(self, dim, data_dim=None, hidden_dim=512, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if data_dim is not None:
            self.attention = Attention(dim, data_dim, hidden_dim, num_heads)
        else:
            self.attention = Attention(dim, dim, hidden_dim, num_heads)
        # self.out = nn.Linear(hidden_dim, dim)


    def forward(self, x, d=None, mask=None):
        if d is not None:
            x = self.norm(x + self.attention(x, d, mask))
        else:
            x = self.norm(x + self.attention(x, x, mask))
        # att_out = self.out(att_out)
        
        return x
    
class PostNormFFN(nn.Module):
    """
    一个带残差和 LayerNorm 的前馈网络，
    用作 Transformer 中的 FFN sub‐layer。
    """
    def __init__(self, dim, hidden_dim=None, act='gelu', dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        # 如果没给 hidden_dim，就用 4*dim 作为 FFN 隐藏层大小
        hd = hidden_dim if hidden_dim is not None else dim * 4
        # 两层 MLP: dim -> hd -> dim
        self.ffn = BasicMLP([dim, hd, dim], act=act, dropout=dropout)

    def forward(self, x):
        # x + ffn(x) 然后 LayerNorm
        return self.norm(x + self.ffn(x))

    
