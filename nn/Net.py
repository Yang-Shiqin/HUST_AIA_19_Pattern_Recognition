#!/usr/bin/env python
# coding=utf-8
import torch.nn as nn
import matplotlib.pyplot as plt

class LinearBlock(nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.net(x)

class MulNet(nn.Module):
    # dim_in: 数据维数
    # C: 分类数
    # dim_h: 隐藏层神经元个数
    # 至少2层
    def __init__(self, dim_in, C, dim_h=8, num_blocks=2, block=LinearBlock) -> None:
        super().__init__()
        blocks = [block(dim_in, dim_h)]
        for _ in range(num_blocks-2):
            blocks.append(block(dim_h, dim_h))
        blocks.append(block(dim_h, C))
        self.net = nn.Sequential(
            nn.Flatten(),
            *blocks,
            nn.Softmax()
        )
        # 初始化策略
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
