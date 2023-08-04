# -*- coding: utf-8 -*-
# @Time    : 2021/6/15 19:55
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.7.9

import torch
from torch import nn, optim

from net.LeNet import LeNet
from utils.LoadData import LoadToNumpy


# =====================================================
# ---------设备选择-------------
if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')
# ----------------------------

TRAIN = True
VALID = True

# ---------超参数设置-----------
LR_RATE = 1e-3
STOP_EPS = 1e-4
EPOCH = 20
BATCH_SIZE = 100
TRAIN_SIZE = 50000
# ----------------------------

# ----------visdom绘图---------
VIZ_SHOW = False
# ----------------------------

# 可视化训练, 本质是web服务器,需要提前开启  python -m visdom.server
if VIZ_SHOW:
    from visdom import Visdom
    viz = Visdom()
    viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
    viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.', legend=['loss', 'acc.']))
# =====================================================


def TrainModel(model, criteon, optimizer):
    """
    Args:
        model:神经网络模型
        criteon:损失函数
        optimizer:优化器

    Returns:
        None
    """
    # 训练过程
    model.to(DEVICE)
    criteon.to(DEVICE)
    train_num = int(TRAIN_SIZE / BATCH_SIZE)  # train共6w张,但是训练(网络参数)前5w张,留1w验证,留1w测试最好
    bestFitness = 0
    for currentEpoch in range(EPOCH):
        # 设置模型训练模式
        model.train()
        print("开始第{%d/%d}次训练纪元." % (currentEpoch + 1, EPOCH))
        for currentBatch in range(train_num):
            # data: [100, 784]    target: [100]
            data, target = trainImg[currentBatch * BATCH_SIZE: (currentBatch + 1) * BATCH_SIZE].to(DEVICE), \
                           trainLabel[currentBatch * BATCH_SIZE: (currentBatch + 1) * BATCH_SIZE].to(DEVICE)

            # 经过神经网络的预测数据
            logits = model.forward(data)  # pred: [100, 10]  <-- [batch_size, out_channel]

            # 损失误差计算
            loss = criteon(logits, target.long())
            if currentBatch % 50 == 0:
                print("Batch id:{: >4d}, 训练误差(交叉熵):{}".format(currentBatch, str(loss)))

            optimizer.zero_grad()      # 清空梯度信息(防止累加梯度)
            loss.backward()            # 梯度向后传播
            optimizer.step()           # 计算图更新
        # end all batches

        # ---------验证部分--------
        # 设置模型验证模式
        model.eval()
        valid_loss = 0

        data, target = trainImg[TRAIN_SIZE:60000].to(DEVICE), trainLabel[TRAIN_SIZE:60000].to(DEVICE)
        pred = model(data)

        pred_number = pred.argmax(dim=1)
        valid_loss += criteon(pred, target.long()).item()
        correct = int(pred_number.eq(target.long()).sum().item()) / (60000-TRAIN_SIZE)

        print("Epoch id:{:0>4d}, 验证误差(交叉熵):{}, 验证准确率:{:.2%}".format(currentEpoch, str(valid_loss), correct))
        if correct >= bestFitness:
            if correct-bestFitness <= STOP_EPS:
                print("验证正确率收敛,提前退出训练Epoch")
                break
            bestFitness = correct

        # 可视化验证部分
        if VIZ_SHOW:
            viz.line([[valid_loss, correct / 10000]], [currentEpoch + 1], win='test', update='append')
            # viz.images(data.view(-1, 1, 28, 28), win='x')
            viz.images(data.view(-1, 1, 28, 28)[0:100], win='x')
            viz.text(str(pred[0:100].detach().cpu().numpy()), win='pred', opts=dict(title='pred'))
    # end Epoch
    torch.save(model.state_dict(), 'net.params')


def Evaluation(model, params):
    """

    Args:
        model:神经网络模型
        params:模型保存参数的路径

    Returns:
        None
    """
    try:
        model.load_state_dict(torch.load(params))
    except Exception as e:
        print(e)
    model.to(DEVICE)
    model.eval()

    data, target = testImg.to(DEVICE), testLabel.to(DEVICE)
    pred = model(data)

    pred_number = pred.argmax(dim=1)
    criteon = nn.CrossEntropyLoss()
    test_loss = criteon(pred, target.long()).item()
    correct = int(pred_number.eq(target.long()).sum().item())
    print("测试误差(交叉熵):{}, 测试准确率:{:.2%}".format(str(test_loss), correct / 10000))


if __name__ == '__main__':
    # 数据加载为numpy类型
    trainImg, testImg, trainLabel, testLabel = LoadToNumpy("./MNIST_DATA", "all")

    # numpy数据转换到tensor类型（暂存内存）
    trainImg = torch.from_numpy(trainImg).float().reshape(-1, 28*28)
    testImg = torch.from_numpy(testImg).float().reshape(-1, 28*28)
    trainLabel = torch.from_numpy(trainLabel).float()
    testLabel = torch.from_numpy(testLabel).float()

    # 神经网络的训练和评估
    if TRAIN:
        # 设置神经网络
        net = LeNet()

        # 设置损失函数为交叉熵方式
        setCriteon = nn.CrossEntropyLoss()

        # 设置优化求解器
        setOptimizer = optim.Adam(net.parameters(), lr=LR_RATE)
        # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)

        # 开始训练
        TrainModel(net, setCriteon, setOptimizer)

    if VALID:
        # 模型重载，预测评估
        clone = LeNet()
        Evaluation(clone, "net.params")
