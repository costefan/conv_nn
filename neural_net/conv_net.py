from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from neural_net.utils import diff_mse

from neural_net.layers import (
    conv_scalar, conv_vector,
    relu_scalar, relu_vector,
    max_pool_scalar, max_pool_vector,
    fc_scalar, fc_vector
)


class SimpleConvNet(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.conv_layer = nn.Conv2d(in_channels=1,
                                    out_channels=20,
                                    kernel_size=5,
                                    stride=1,
                                    padding=0,
                                    dilation=1,
                                    groups=1,
                                    bias=True)
        self.fc_layer1 = nn.Linear(in_features=20 * 12 * 12, out_features=500)
        self.fc_layer2 = nn.Linear(in_features=500, out_features=10)

        self.to(device)

    def forward_pytorch(self, x):
        # When your implementations will be ready, replace standard Pytorch
        # implementation by your custom functions, like:
        #
        # z_conv = conv2d_vector(x, conv_weight=self.conv_layer.weight,
        #                       conv_bias=self.conv_layer.bias,
        #                       device=self.device)
        # self.conv_layer(x)
        z_conv = self.conv_layer(x,)

        z_pool = F.max_pool2d(z_conv, 2, 2)
        z_pool_reshaped = z_pool.view(-1, 20 * 12 * 12)
        z_fc1 = self.fc_layer1(z_pool_reshaped)
        z_relu = F.relu(z_fc1)
        z_fc2 = self.fc_layer2(z_relu)
        y = F.softmax(z_fc2, dim=1)

        return y

    def forward_scalar(self, x):
        z_conv = conv_scalar(x,
                             conv_weight=self.conv_layer.weight,
                             conv_bias=self.conv_layer.bias,
                             device=self.device,
                             layer_config={
                                 'stride': 1,
                                 'padding': 0
                             })
        z_pool = max_pool_scalar(z_conv,
                                 device=self.device,
                                 layer_config={'stride_x': 2,
                                               'stride_y': 2},)
        z_pool_reshaped = z_pool.view(-1, 20 * 12 * 12)

        z_fc1 = fc_scalar(z_pool_reshaped,
                          weight=self.fc_layer1.weight,
                          bias=self.fc_layer1.bias,
                          device=self.device)

        z_fc1_ = self.fc_layer1(z_pool_reshaped)

        print(z_fc1_.shape)
        print(z_fc1.shape)
        print(diff_mse(z_fc1_, z_fc1))

        z_relu = relu_scalar(z_fc1, self.device)
        z_fc2 = self.fc_layer2(z_relu)
        y = F.softmax(z_fc2, dim=1)

        return y

    def forward_vectorized(self):
        pass

    def forward(self, x):
        return self.forward_scalar(x)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(torch.log(output), target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
