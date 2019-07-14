from .base import BaseLayer

import torch


def fc_vector(a, weight, bias, device):
    pass


def fc_scalar(a, weight, bias, device):
    batch_size, _ = a.shape
    H, W = weight.shape

    result = torch.empty(batch_size, H)

    for n in range(batch_size):
        for j in range(H):
            for i in range(W):
                result[n, j] += weight[j, i] * a[n, i] + bias[j]

    return result.to(device)
