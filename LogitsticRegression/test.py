#!/usr/bin/env python
# coding=utf-8
import torch
import random
from LogisticRegression import LogisticRegression

pos_x = torch.randn(200, 2)+torch.tensor([-5, 0])
neg_x = torch.randn(200, 2)+torch.tensor([0, 5])
x = torch.cat((pos_x, neg_x), dim=0)
pos_y = torch.ones(200,)
neg_y = -torch.ones(200,)
y = torch.cat((pos_y, neg_y), dim=0)
# x = x.numpy().tolist()
# y = y.numpy().tolist()
# tmp = list(zip(x, y))
# random.shuffle(tmp)
# x, y = zip(*tmp)
# x = torch.tensor(x)
# y = torch.tensor(y)
N, d = pos_x.shape
idx = [i for i in range(2*N)]
random.shuffle(idx)
train_x, train_y = x[idx[:int(2*N*0.8)]], y[idx[:int(2*N*0.8)]]
test_x, test_y = x[idx[int(2*N*0.8):]], y[idx[int(2*N*0.8):]]
model = LogisticRegression(d)
print(model.train(train_x, train_y, method='M', it=100))    # w
model.draw(train_x, train_y, test_x, test_y)
print("Probability of being a positive class: \n", torch.cat((test_x, model.predict(test_x).reshape(-1, 1)), dim=1))