from .base import BaseLayer


def conv_vector(x_in, conv_weight, conv_bias, device):
    # Let this be 3x3 convolution with stride = 1 and padding = 1
    # # Suppose our X is 5x1x10x10, X_col will be a 9x500 matrix
    # X_col = im2col_indices(X, h_filter, w_filter, padding=padding,
    #                        stride=stride)
    # # Suppose we have 20 of 3x3 filter: 20x1x3x3. W_col will be 20x9 matrix
    # W_col = W.reshape(n_filters, -1)
    #
    # # 20x9 x 9x500 = 20x500
    # out = W_col @ X_col + b
    #
    # # Reshape back from 20x500 to 5x20x10x10
    # # i.e. for each of our 5 images, we have 20 results with size of 10x10
    # out = out.reshape(n_filters, h_out, w_out, n_x)
    # out = out.transpose(3, 0, 1, 2)
    pass


def conv_scalar(x_in, conv_weight, conv_bias, device):
    print(type(x_in))
    print(type(conv_weight))
    print(type(conv_bias))
    return