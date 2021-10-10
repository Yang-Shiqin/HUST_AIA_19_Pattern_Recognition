#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import DataLoader, sampler

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

    def forward(self, x):
        score = self.net(x)
        return score

    def inferrence(self):
        pass

def train(model, optim):
    x,y = xxx
    loss = F.cross_entropy(model(x), y)





def load_mnist(path='./mnist', num_train=49000, num_val=1000, batch_size=64):
    transform = T.Compose([
                  T.ToTensor(),
                  T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
              ])
    
    mnist_train = dset.MNIST(path, train=True, download=True, transform=transform)
    loader_train = DataLoader(mnist_train, batch_size, sampler=sampler.SubsetRandomSampler(range(num_train)))
    
    mnist_val = dset.MNIST(path, train=True, download=True, transform=transform)
    loader_val = DataLoader(mnist_val, batch_size, sampler=sampler.SubsetRandomSampler(range(num_train, num_train+num_val)))
    
    mnist_test = dset.CIFAR10(path, train=False, download=True, transform=transform)
    loader_test = DataLoader(mnist_test, batch_size)
    
    return loader_train, loader_val, loader_test