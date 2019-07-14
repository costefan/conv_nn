from .base import BaseLayer

import torch


def relu_vector(self, a, device):
    pass


def relu_scalar(a, device):
    X_batch, el_size = a.shape
    res = torch.zeros(a.shape)

    for img in range(X_batch):
        for el in range(el_size):
            res[img, el] = max(0, a[img, el])

    return res.to(device)
