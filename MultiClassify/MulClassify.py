#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from helper import load_mnist, train

plt.rcParams['image.cmap'] = 'gray'

class SoftClassify(nn.Module):
    # dim_in: 数据维数
    # C: 分类数
    def __init__(self, dim_in, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim_in, C),
            nn.ReLU(),
            nn.Softmax()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight)
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        score = self.net(x)
        return score



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



################## test ##################
if __name__ == '__main__':
    loader_train, loader_test = load_mnist(batch_size=256)  # 64效果比256好很多

    dim_in = 784
    C = 10
    learning_rate = 0.1
    momentum = 0.5

    model = SoftClassify(dim_in, C)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
    #                       momentum=momentum, nesterov=True)

    acc_history, acc_test_history, loss_history, iter_history = train(model, loader_train, loader_test, optimizer, epoch=10)
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
    show_mnist_sample(model, loader_test)