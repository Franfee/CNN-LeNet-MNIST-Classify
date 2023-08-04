# -*- coding: utf-8 -*-
# @Time    : 2021/6/15 19:56
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.7.9
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):

    def __init__(self):
        super(Linear, self).__init__()
        # xw+b
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x_in):
        # x: [b, 1, 28, 28]
        # h1 = relu(xw1+b1)
        x_out = F.relu(self.fc1(x_in))
        # h2 = relu(h1w2+b2)
        x_out = F.relu(self.fc2(x_out))
        # h3 = h2w3+b3
        x_out = self.fc3(x_out)
        return x_out


if __name__ == '__main__':
    net = Linear()

    a = torch.rand(1, 784, dtype=torch.float)
    print(net(a))

    b = torch.rand(2, 784, dtype=torch.float)
    print(net(b))