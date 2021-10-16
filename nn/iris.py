#!/usr/bin/env python
# coding=utf-8
import torch
import torch.optim as optim
import random
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader

from Net import MulNet, myshow
from helper import train

class IrisDataset(torch.utils.data.Dataset):
    def __init__(self, data_X, data_y, train=False):
        self.data_X = data_X
        self.data_y = data_y
        self.train = train

    def __getitem__(self, index):
        return self.data_X[index], self.data_y[index]

    def __len__(self):
        return self.data_X.shape[0]


############ test ################
if __name__ == '__main__':
    # 得到样本
    data = load_iris()
    x = torch.tensor(data.data)
    y = torch.tensor(data.target)

    # shuffle
    x = x.numpy().tolist()
    y = y.numpy().tolist()
    tmp = list(zip(x, y))
    random.shuffle(tmp)
    x, y = zip(*tmp)
    x = torch.tensor(x)
    y = torch.tensor(y)

    # 抽样
    train_num = 30
    test_num = 20
    train_x = x[:train_num]
    train_y = y[:train_num]
    test_x = x[train_num:train_num+test_num]
    test_y = y[train_num:train_num+test_num]


    # nn
    model = MulNet(4, 3, num_blocks=2)
    # optimizer = optim.SGD(model.parameters(), lr=0.5)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5, nesterov=True)

    # isir生成dataset
    train_dataset = IrisDataset(train_x, train_y, True)
    test_dataset = IrisDataset(test_x, test_y)
    loader_train = DataLoader(train_dataset, 30)
    loader_test = DataLoader(test_dataset, 20)

    acc_history, acc_test_history, loss_history, iter_history = train(model, loader_train, loader_test, optimizer, epoch=30)
    myshow(acc_history, acc_test_history, loss_history, iter_history)

    