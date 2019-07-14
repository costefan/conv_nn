from .base import BaseLayer

import torch


def fc_vector(a, weight, bias, device):
    pass


def fc_scalar(a, weight, bias, device):
    print('Input image dims: {}'.format(a.shape))
    print('Weight dims: {}'.format(weight.shape))
    print('Bias dims: {}'.format(bias.shape))
    X_batch, X_size = a.shape
    W_h, W_w = weight.shape

    res = torch.zeros(X_batch, W_h)

    for img in range(X_batch):
        for w_i in range(W_h):
            for w_j in range(W_w):
                res[img, w_i] += weight[w_i, w_j] * a[img, w_j] + bias[w_i]

    return res.to(device)
