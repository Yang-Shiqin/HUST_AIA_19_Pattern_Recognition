#!/usr/bin/env python
# coding=utf-8
import torch

def softmax(x):
    ex = torch.exp(x)
    return ex/ex.sum(dim=1).reshape(-1,1)


class SoftMaxClassify:
    def __init__(self, dim_in, C):
        self.w = torch.zeros(dim_in+1, C)

    def predict(self, x):
        N, d = x.shape
        x = torch.cat((torch.ones(N,1), x), dim=1)
        return softmax(x@self.w)

    # x(1, d)
    # y(1, C)
    def _dw(self, x, y):
        N, d = x.shape
        _, C = y.shape
        k = torch.argmax(y) # 正确类别
        xx = torch.cat((torch.ones(N,1), x), dim=1)
        dw = torch.zeros(d+1, C)
        for i in range(C):
            y_pre = self.predict(x).squeeze()
            dw[:, i] += (((y_pre[i]-1)*xx.reshape(-1,1)) if i==k else (y_pre[i]*xx.reshape(-1,1))).reshape(-1)
        return dw

    def train(self, x, y):
        it = 0
        N, _ = x.shape
        _, C = self.w.shape
        # print(torch.argmax(self.predict(x), dim=1))
        # print(torch.argmax(y, dim=1))
        while (torch.argmax(self.predict(x), dim=1)!=torch.argmax(y, dim=1)).int().sum()!=0:
            y_pre = self.predict(x[it%N].unsqueeze(0))
            print("w: ", self.w)
            print("y: ", y_pre)
            self.w -= self._dw(x[it%N].unsqueeze(0), y[it%N].unsqueeze(0))
            it += 1
        return self.w


x = torch.tensor([  [3,0],
                    [3,6],
                    [0,3],
                    [-3,0]])
y = torch.tensor([  [1,0,0],
                    [1,0,0],
                    [0,1,0],
                    [0,0,1]])


dim_in = x.shape[1]
C = y.shape[1]
model = SoftMaxClassify(dim_in, C)
w = model.train(x, y)
model.predict(x)