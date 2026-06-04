import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def q_error_criterion(true_card, est_card):
    batch_ones = torch.ones(true_card.shape, dtype=torch.float32).to(est_card.device)
    fixed_true_cards = torch.where(true_card == 0., batch_ones, true_card)
    fixed_est_cards = torch.where(est_card == 0., batch_ones, est_card)
    q_error = torch.where(fixed_true_cards > fixed_est_cards, fixed_true_cards / fixed_est_cards,
                          fixed_est_cards / fixed_true_cards)

    return q_error


def weight_q_error(true_card, est_card):
    # weight = true_card / torch.mean(true_card)
    # weight = torch.clip(weight, 1e-3, torch.max(weight))
    weight = torch.log(true_card) / torch.sum(torch.log(true_card))

    batch_ones = torch.ones(true_card.shape, dtype=torch.float32).to(true_card.device)
    fixed_true_cards = torch.where(true_card == 0., batch_ones, true_card)
    fixed_est_cards = torch.where(est_card == 0., batch_ones, est_card)
    q_error = torch.where(fixed_true_cards > fixed_est_cards, fixed_true_cards / fixed_est_cards,
                          fixed_est_cards / fixed_true_cards)
    # print(q_error)

    return q_error * weight


def weight_mse_loss(true_card, est_card):
    # weight = true_card / torch.mean(true_card)
    # weight = torch.clip(weight, 1e-3, torch.max(weight))
    weight = torch.log(true_card) / torch.sum(torch.log(true_card))

    return weight * ((true_card - est_card) ** 2)


def recounstruction_error_criterion(sm, train_out, nk, node_array, device, lambd=1e-4):
    # MSE loss - recoustructed matrix vs original
    keyword_node = np.array([ki for ki, kv in enumerate(node_array) if kv < nk])
    object_node = np.array([oi for oi, ov in enumerate(node_array) if ov >= nk])

    o_emb = torch.transpose(train_out[object_node], 0, 1)
    k_emb = train_out[keyword_node]
    true_mat = sm[node_array[keyword_node], :][:, node_array[object_node] - nk]
    gt = torch.Tensor(true_mat.todense()).float().to(device)
    
    recounstructed_loss = F.mse_loss(gt, k_emb @ o_emb) # @ - torch.matmul()

    # regularization loss
    regularization_loss = torch.norm(k_emb, 2) + torch.norm(o_emb, 2)

    return recounstructed_loss + lambd * regularization_loss


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers.to(L2_distances.device))[:, None, None]).sum(dim=0)


def mmd_loss(x, y, width=1):
    x_n = x.shape[0]
    y_n = y.shape[0]

    x_square = torch.sum(x * x, 1)
    y_square = torch.sum(y * y, 1)

    kxy = torch.matmul(x, y.t())
    kxy = kxy - 0.5 * x_square.unsqueeze(1).expand(x_n, y_n)
    kxy = kxy - 0.5 * y_square.expand(x_n, y_n)
    kxy = torch.exp(width * kxy).sum() / x_n / y_n

    kxx = torch.matmul(x, x.t())
    kxx = kxx - 0.5 * x_square.expand(x_n, x_n)
    kxx = kxx - 0.5 * x_square.expand(x_n, x_n)
    kxx = torch.exp(width * kxx).sum() / x_n / x_n

    kyy = torch.matmul(y, y.t())
    kyy = kyy - 0.5 * y_square.expand(y_n, y_n)
    kyy = kyy - 0.5 * y_square.expand(y_n, y_n)
    kyy = torch.exp(width * kyy).sum() / y_n / y_n

    return kxx + kyy - 2 * kxy


class MMD_Loss(nn.Module):
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY
    

class Regularizer(nn.Module):
    def __init__(self):
        super(Regularizer, self).__init__()
    

    def forward(self, tuples):
        raise NotImplementedError


class ModelRegularizer(Regularizer):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight


    def forward(self, model):
        norm = 0
        for param in model.parameters():
            norm += torch.norm(param, 2)
        return self.weight * norm


class L2(Regularizer):
    def __init__(self, weight):
        super(L2, self).__init__()
        self.weight = weight

    def forward(self, tuples):
        norm = 0
        for f in tuples:
            norm += torch.norm(f, p=2)
        return self.weight * norm


