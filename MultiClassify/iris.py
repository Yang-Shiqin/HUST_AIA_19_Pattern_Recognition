#!/usr/bin/env python
# coding=utf-8
import torch
import torch.optim as optim
import random
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader

from Pocket import Pocket
from MulClassify import SoftClassify
from helper import train


class ovo():
    def __init__(self, C):
        self.C = C
        self.w = []
        self.w_num = int(C*(C-1)/2)

    def train(self, x, y):
        for i in range(self.C):
            for j in range(i):  # 下三角
                xp = x[y==i]
                xn = x[y==j]
                xx = torch.cat((xp, xn), dim=0)
                yy = torch.ones(sum(y==i))
                yy = torch.cat((yy, -torch.ones(sum(y==j))))
                model = Pocket(xx, yy)
                w, _ = model.result(100, is_print='True', is_draw='False')
                self.w.append(w.numpy())
        self.w = torch.tensor(self.w).t()
        return self.w

    def predict(self, x, y, test_x=None, test_y=None):
        # vote = torch.sign(x@self.w)
        N, d = x.shape
        x = torch.cat((torch.ones(N,1), x), dim=1)
        vote = torch.zeros(N, self.C)
        if test_x!=None:
            test_x = torch.cat((torch.ones(test_x.shape[0],1), test_x), dim=1)
            vote_test = torch.zeros(test_x.shape[0], self.C)
        for i in range(1,self.C):
            for j in range(i):
                tmp = x@self.w
                vote[:, i] += (tmp[:, int(i*(i-1)/2+j)])>0
                vote[:, j] += (tmp[:, int(i*(i-1)/2+j)])<0
                if test_x!=None:
                    tmp = test_x@self.w
                    vote_test[:, i] += (tmp[:, int(i*(i-1)/2+j)])>0
                    vote_test[:, j] += (tmp[:, int(i*(i-1)/2+j)])<0
        pre_y = torch.argmax(vote, dim=1).int()
        # print(pre_y)
        # print(y)
        rate = sum(pre_y==y)/y.shape[0]
        print("ovo train correct rate: {}".format(rate))
        # print(torch.sign(x@self.w))
        # print(vote)
        if test_x!=None:
            pre_test_y = torch.argmax(vote_test, dim=1).int()
            test_rate = sum(pre_test_y==test_y)/test_y.shape[0]
            print("ovo test correct rate: {}".format(test_rate))
        

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

    # ovo
    model = ovo(3)
    model.train(train_x, train_y)
    # model.predict(test_x, test_y)
    model.predict(train_x, train_y, test_x, test_y)

    # softmax
    model = SoftClassify(4, 3)
    # optimizer = optim.SGD(model.parameters(), lr=0.5)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5, nesterov=True)

    # isir生成dataset
    train_dataset = IrisDataset(train_x, train_y, True)
    test_dataset = IrisDataset(test_x, test_y)
    loader_train = DataLoader(train_dataset, 30)
    loader_test = DataLoader(test_dataset, 20)

    acc_history, acc_test_history, loss_history, iter_history = train(model, loader_train, loader_test, optimizer, epoch=30)
    print(acc_history[-1], acc_test_history[-1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax.plot(iter_history, acc_history, 'r-', label='train acc')
    ax.plot(iter_history, acc_test_history, 'b-', label='test acc')
    ax2.plot(iter_history, loss_history, 'c-', label='loss')
    fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    ax.set_ylabel(r"acc")
    ax2.set_ylabel(r"loss")
    plt.show()
    