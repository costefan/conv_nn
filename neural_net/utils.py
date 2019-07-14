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


def im_2_col(img, K_h, device, stride=1, padding=0):
    batch_size, C_in, X_h, X_w = img.shape

    out_size = ((X_h - K_h + 2 * padding) // stride) + 1

    col = torch.zeros((K_h, K_h, batch_size, C_in, out_size, out_size))

    margin = stride * out_size

    for x in range(K_h):
        for y in range(K_h):
            col[x, y] = img[:, :, x:x + margin:stride, y:y + margin:stride]

    res = col.view(K_h * K_h, -1)

    return res.to(device)