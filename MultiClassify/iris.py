#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.functional as F
import random
from sklearn.datasets import load_iris
from Pocket import Pocket


class ovo():
    def __init__(self, C):
        for i in range(C):
            for j in range(i):  # 上三角
                pass
        

############ test ################
if __name__ == '__main__':
    data = load_iris()
    x = data.data
    y = data.target

    x = x.numpy().tolist()
    y = y.numpy().tolist()
    tmp = list(zip(x, y))
    random.shuffle(tmp)
    x, y = zip(*tmp)
    x = torch.tensor(x)
    y = torch.tensor(y)

    train_num = 30
    test_num = 20
    train_x = x[:train_num]
    train_y = y[:train_num]
    test_x = x[train_num:train_num+test_num]
    test_y = y[train_num:train_num+test_num]

