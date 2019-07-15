from .base import BaseLayer

import torch
import numpy as np
from neural_net.utils import diff_mse


def fc_vector(a, weight, bias, device):
    res = a.mm(weight.t()) + bias

    return res.to(device)


def fc_scalar(a, weight, bias, device):
    batch_size, _ = a.shape
    H, W = weight.shape

    result = torch.zeros(batch_size, H)

    for n in range(batch_size):
        for j in range(H):
            for i in range(W):
                result[n, j] += weight[j, i] * a[n, i]

            result[n, j] += bias[j]

    return result.to(device)
