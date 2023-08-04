# -*- coding: utf-8 -*-
# @Time    : 2021/6/15 19:55
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.7.9

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)


class LeNet(nn.Module):
    """
    输入维度(input dim): (N,784)
    输出维度(output dim): (N,10)
    在图片尺寸比较大时，LeNet在图像分类任务上存在局限性。
    """
    def __init__(self):
        super(LeNet, self).__init__()

        self.model = nn.Sequential(
            Reshape(),
            nn.Conv2d(1, 6, kernel_size=(5, 5), padding=(2, 2)),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x_in):
        x_out = self.model(x_in)
        return x_out


if __name__ == '__main__':
    net = LeNet()

    a = torch.rand(1, 784, dtype=torch.float)
    print("a:" + str(a.shape))
    print(net(a))

    b = torch.rand(2, 784, dtype=torch.float)
    print("b:" + str(b.shape))
    print(net(b))

    c = torch.rand(3, 784, dtype=torch.float)
    print("c:" + str(c.shape))
    print(net(b))
