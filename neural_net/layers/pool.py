from .base import BaseLayer
import torch


def max_pool_vector(a, device, layer_config: dict):
    X_batch, filters_num, X_h, X_w = a.shape
    output_dim_x = X_h // layer_config['stride_x']
    output_dim_y = X_h // layer_config['stride_y']


def max_pool_scalar(a, device, layer_config: dict):
    X_batch, filters_num, X_h, X_w = a.shape
    output_dim_x = X_h // layer_config['stride_x']
    output_dim_y = X_h // layer_config['stride_y']
    res = torch.zeros(X_batch, filters_num,
                      output_dim_x,
                      output_dim_y)

    for img in range(X_batch):
        for filter_ix in range(filters_num):
            for i in range(output_dim_x):
                for j in range(output_dim_y):
                    res[img, filter_ix, i, j] = \
                        torch.max(a[img, filter_ix,
                                  i*layer_config['stride_x']:i*layer_config['stride_x']+layer_config['stride_x'],
                                  j*layer_config['stride_y']:j*layer_config['stride_y']+layer_config['stride_y']])

    return res.to(device)
