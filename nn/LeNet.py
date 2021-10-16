#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from helper import load_mnist, train

plt.rcParams['image.cmap'] = 'gray'


class LeNet(nn.Module):
    # dim_in: 数据维数
    # C: 分类数
    # dim_h: 隐藏层神经元个数
    # 至少2层
    def __init__(self, C, H, W) -> None:
        super().__init__()
        assert(C==1)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(int(16*(H/4-2)*(W/4-2)), 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
            nn.Softmax()
        )
        # 初始化策略
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        score = self.net(x)
        return score


# 抽样展示预测效果
def show_mnist_sample(model, loader, batch_size=64, num=10, row=5):
    if loader.dataset.train:
        print('show random sample on train set')
    else:
        print('show random sample on test set')
    full_num = batch_size
    sample = list(torch.utils.data.WeightedRandomSampler(torch.ones(full_num)/full_num,num_samples=num, replacement=False))
    examples = enumerate(loader)
    _, (example_data, example_targets) = next(examples) # 实际上只是在第一个batch里抽样
    fig = plt.figure(figsize=(8, 6))
    for i in range(num):
        plt.subplot(math.ceil(num/row), row, i+1)
        # plt.tight_layout()
        plt.imshow(example_data[sample[i]][0], cmap='gray', interpolation='none')
        plt.title("Pre: {}, act: {}".format((model(example_data)[sample[i]]).argmax(), example_targets[sample[i]]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def myshow(acc_history, acc_test_history, loss_history, iter_history):
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

################## test ##################
if __name__ == '__main__':
    loader_train, loader_test = load_mnist(batch_size=256)  # 64效果比256好很多
    _, H, W = loader_train.dataset.data.shape

    learning_rate = 0.2
    momentum = 0.5

    model = LeNet(1, H, W)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                          momentum=momentum, nesterov=True)

    acc_history, acc_test_history, loss_history, iter_history = train(model, loader_train, loader_test, optimizer, epoch=10)
    myshow(acc_history, acc_test_history, loss_history, iter_history)
    show_mnist_sample(model, loader_test)