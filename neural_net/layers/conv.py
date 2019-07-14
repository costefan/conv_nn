from .base import BaseLayer
import torch

# from neural_net.utils import im2col_indices


def calculate_output_conv_size(img_size, filter_size, padding, stride):
    """Calculate size of the tenzor after convolution
    only for quadratic matrices case
    (img_size - filter_size + 2 * padding) / stride + 1
    :return: 
    """
    return (img_size - filter_size + 2 * padding) // stride + 1


def conv_vector(x_in, conv_weight, conv_bias, device, layer_config: dict):
    # X_batch, C_num, X_h, X_w = x_in.shape
    # filters_size, _, K_h, K_w = conv_weight.shape
    # output_conv_size = int(calculate_output_conv_size(X_h, K_h,
    #                                               layer_config['padding'],
    #                                               layer_config['stride']))
    # # Let this be 3x3 convolution with stride = 1 and padding = 1
    # # Suppose our X is 5x1x10x10, X_col will be a 9x500 matrix
    # X_col = im2col_indices(x_in, X_h, X_w,
    #                        padding=layer_config['padding'],
    #                        stride=layer_config['stride'])
    # # Suppose we have 20 of 3x3 filter: 20x1x3x3. W_col will be 20x9 matrix
    # W_col = conv_weight.reshape(filters_size, -1)
    #
    # # 20x9 x 9x500 = 20x500
    # out = W_col @ X_col + conv_bias
    #
    # # Reshape back from 20x500 to 5x20x10x10
    # # i.e. for each of our 5 images, we have 20 results with size of 10x10
    # out = out.reshape(filters_size, h_out, w_out, n_x)
    # out = out.transpose(3, 0, 1, 2)
    # pass
    pass


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
