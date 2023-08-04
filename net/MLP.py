# -*- coding: utf-8 -*-
# @Time    : 2021/6/15 20:00
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.7.9
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    输入维度(input dim): (N,784)
    输出维度(output dim): (N,10)
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),)

    def forward(self, x_in):
        x_out = self.model(x_in)
        return x_out


if __name__ == '__main__':
    net = MLP()

    a = torch.rand(1, 784, dtype=torch.float)
    print("a:" + str(a.shape))
    print(net(a))

    b = torch.rand(2, 784, dtype=torch.float)
    print("b:" + str(b.shape))
    print(net(b))

    c = torch.rand(3, 784, dtype=torch.float)
    print("c:" + str(c.shape))
    print(net(b))
