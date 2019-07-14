from .base import BaseLayer
import torch

from neural_net.utils import im_2_col


def calculate_output_conv_size(img_size, filter_size, padding, stride):
    """Calculate size of the tenzor after convolution
    only for quadratic matrices case
    (img_size - filter_size + 2 * padding) / stride + 1
    :return: 
    """
    return (img_size - filter_size + 2 * padding) // stride + 1


def conv_vector(x_in, conv_weight, conv_bias, device, layer_config: dict):

    X_batch, C_num, X_h, X_w = x_in.shape
    filters_size, _, K_h, K_w = conv_weight.shape
    output_conv_size = int(calculate_output_conv_size(X_h, K_h,
                                                      layer_config['padding'],
                                                      layer_config['stride']))

    x_col = im_2_col(x_in, K_h, device,
                     layer_config['stride'],
                     layer_config['padding'])

    w_row = conv_weight.reshape(filters_size, -1)

    res = w_row.mm(x_col).add(conv_bias.view(filters_size, 1))
    res = res.view(filters_size, X_batch, C_num,
                   output_conv_size, output_conv_size).sum(dim=2)
    res = res.transpose(0, 1)

    return res.to(device)


def conv_scalar(x_in, conv_weight, conv_bias, device, layer_config: dict):
    X_batch, C_num, X_h, X_w = x_in.shape
    filters_size, _, K_h, K_w = conv_weight.shape
    output_conv_size = int(calculate_output_conv_size(X_h, K_h,
                                                      layer_config['padding'],
                                                      layer_config['stride']))

    print('Input image dims: {}'.format(x_in.shape))
    print('Kernel dims: {}'.format(conv_weight.shape))

    print('Result convolution size: {}'.format(
        (X_batch, filters_size, output_conv_size, output_conv_size)))

    res = torch.empty(X_batch, filters_size,
                      output_conv_size, output_conv_size)

    for batch_img in range(X_batch):
        for filter_ix in range(filters_size):
            for height in range(output_conv_size):
                for width in range(output_conv_size):

                    kernel = conv_weight[filter_ix]
                    bias = conv_bias[filter_ix]

                    img = x_in[batch_img]

                    result = 0
                    for c_in in range(C_num):
                        for i in range(K_h):
                            for j in range(K_w):
                                result += img[
                                    c_in, height + i, width + j
                                ] * kernel[c_in, i, j]

                    res[batch_img, filter_ix, height, width] = \
                        result + bias

    return res.to(device)
