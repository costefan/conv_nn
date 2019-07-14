import torch
import numpy as np

LOG_DIMS = 1


def log_dims(f):
    def wrapper(*args, **kwargs):
        pass
        return f(*args, **kwargs)
    return wrapper


def diff_mse(x, y):
    x_vec = x.view(1, -1).squeeze()
    y_vec = y.view(1, -1).squeeze()

    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()


