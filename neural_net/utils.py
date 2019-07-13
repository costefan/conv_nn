import torch
import numpy as np


def diff_mse(x, y):
    x_vec = x.view(1, -1).squeeze()
    y_vec = y.view(1, -1).squeeze()

    return torch.mean(torch.pow((x_vec - y_vec), 2)).item()


def im2col_indices(D, F1, F2, H_, W_, S):
    """
    Input:
    - D:  size of depth
    - F1: height of convolution
    - F2: width of convolution
    - H_: height of activation maps
    - W_: width of activation maps 
    - S:  stride of convolution
    Output:
    - d:[(D*F1*F2*H_*W_) * 1] -> indices for depth
    - h:[(D*F1*F2*H_*W_) * 1] -> indices for height
    - w:[(D*F1*F2*H_*W_) * 1] -> indices for width
    """
    d = np.array([d_
        for d_ in range(D)
        for f1 in range(F1)
        for f2 in range(F2)
        for h_ in range(H_)
        for w_ in range(W_)
    ])
    h = np.array([h_*S + f1
        for d_ in range(D)
        for f1 in range(F1)
        for f2 in range(F2)
        for h_ in range(H_)
        for w_ in range(W_)
    ])
    w = np.array([w_*S + f2
      for d_ in range(D)
      for f1 in range(F1)
      for f2 in range(F2)
      for h_ in range(H_)
      for w_ in range(W_)
    ])

    return d, h, w


def im2col_forward(X, F1, F2, S, P):
    """
    Helper function for conv_forward & max_pool_forward
    Input:
    - X:  [N * D * H * W] -> images
    - F1: height of convolution
    - F2: width of convolution
    - S:  stride of convolution
    - P:  size of zero padding
    Output:
    - X_col: [(D * F1 * F2) * (H_ * W_) * N] -> column stretched matrix
    """
    N, D, H, W = X.shape
    H_ = (H - F1 + 2*P) / S + 1
    W_ = (W - F2 + 2*P) / S + 1

    # zero-pad X: [D * (H+2P) * (W+2P) * N]
    X_pad = np.pad(X, ((0,0),(0,0),(P,P),(P,P)), 'constant').transpose(1,2,3,0)
    # get indices for X: [(D*F1*F2*H_*W_) * 1]
    d, h, w = im2col_indices(D, F1, F2, H_, W_, S)
    # compute X_col
    X_col = X_pad[d, h, w, :].reshape(D*F1*F2, H_*W_, N)

    return X_col