# -*- coding: utf-8 -*-
# @Time    : 2021/6/15 20:09
# @Author  : FanAnfei
# @Software: PyCharm
# @python  : Python 3.7.9


import struct
import numpy as np
from pathlib import Path


def LoadToNumpy(readPath, mode="all"):
    """
    可以使用pandas来处理csv数据
    Args:
        readPath:读取路径
        mode:["all","train","test"]

    Returns:
        分别返回要求的numpy类型数据
    """
    # --------------------------数据加载------------------------------
    # 数据路径
    dataSetPath = Path(readPath)
    # 训练数据 Path重定义了/
    trainImgPath = dataSetPath / "train-images-idx3-ubyte"
    trainLabelPath = dataSetPath / "train-labels-idx1-ubyte"
    # 测试数据 Path
    testImgPath = dataSetPath / "t10k-images-idx3-ubyte"
    testLabelPath = dataSetPath / "t10k-labels-idx1-ubyte"

    if mode == "all":
        # 读取数据(使用struct 中的unpack,把字节用需要的形式组合.因为有多余的数据)
        with open(trainImgPath, 'rb') as f:
            # 大于号:字节存储方向,大的在前还是小的在前   4: 4个  i:整数    16个bit
            struct.unpack('>4i', f.read(16))
            trainImg = np.fromfile(f, dtype=np.uint8).reshape(-1, 28 * 28) / 255
        with open(testImgPath, 'rb') as f:
            struct.unpack('>4i', f.read(16))
            testImg = np.fromfile(f, dtype=np.uint8).reshape(-1, 28 * 28) / 255
        with open(trainLabelPath, 'rb') as f:
            struct.unpack('>2i', f.read(8))
            trainLabel = np.fromfile(f, dtype=np.uint8)
        with open(testLabelPath, 'rb') as f:
            struct.unpack('>2i', f.read(8))
            testLabel = np.fromfile(f, dtype=np.uint8)
        print("数据读取完成")
        # print(trainImg.shape)  # 60000张 784图
        # print(trainLabel.shape)  # 60000个label
        # print(testImg.shape)  # 10000张 784图
        # print(testLabel.shape)  # 10000个label
        return trainImg, testImg, trainLabel, testLabel
    if mode == "train":
        with open(trainImgPath, 'rb') as f:
            # 大于号:字节存储方向,大的在前还是小的在前   4: 4个  i:整数    16个bit
            struct.unpack('>4i', f.read(16))
            trainImg = np.fromfile(f, dtype=np.uint8).reshape(-1, 28 * 28) / 255
        with open(trainLabelPath, 'rb') as f:
            struct.unpack('>2i', f.read(8))
            trainLabel = np.fromfile(f, dtype=np.uint8)
        return trainImg, trainLabel
    if mode == "test":
        with open(testImgPath, 'rb') as f:
            struct.unpack('>4i', f.read(16))
            testImg = np.fromfile(f, dtype=np.uint8).reshape(-1, 28 * 28) / 255
        with open(testLabelPath, 'rb') as f:
            struct.unpack('>2i', f.read(8))
            testLabel = np.fromfile(f, dtype=np.uint8)
        return testImg, testLabel


if __name__ == '__main__':
    # 数据加载numpy类型
    trainImg1, testImg1, trainLabel1, testLabel1 = LoadToNumpy("../MNIST_DATA")
    print(type(trainImg1))
    print(trainImg1.shape)

    import matplotlib.pyplot as plt
    plt.imshow(trainImg1[0].reshape(28, 28)*255, cmap='gray')
    plt.title("Number: "+str(trainLabel1[0]))
    plt.show()
