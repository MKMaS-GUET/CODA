import torch
import torch.nn as nn
import torch.nn.functional as F


class GEGLU(nn.Module):
    def __init__(self):
        super(GEGLU, self).__init__()
    

    def forward(self, x):
        x, gates = x.chunk(2, -1)
        out = x * F.gelu(gates)
        return out


class FFN(nn.Module):
    def __init__(self, dim, multi=4):
        super(FFN, self).__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * multi * 2), GEGLU(), nn.Linear(dim * multi, dim))
    

    def forward(self, x):
        return self.net(x)
    

class PreNormFFN(nn.Module):
    def __init__(self, dim, multi=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = FFN(dim, multi)
    

    def forward(self, x):
        x = self.norm(x)
        out = self.ffn(x)

        return out


class PostNormFFN(nn.Module):
    def __init__(self, dim, multi=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = FFN(dim, multi)
    

    def forward(self, x):
        x = self.norm(x + self.ffn(x))

        return x
    


