#!/usr/bin/env python
# coding=utf-8
import torch
import matplotlib.pyplot as plt

# x(N, d)
# y(N, )
# w(d, )
# b(N, )
class LinearRegression():
    def __init__(self, N, in_dim):
        self.w = torch.rand(in_dim)*1e-5
        # self.b = torch.zeros(N)
        self.b = 0

    def loss(self, x, y):
        return ((torch.mv(x, self.w)+self.b-y.reshape(-1, 1))**2).sum()

    def backword(self, x, y):
        tmp = torch.mv(x, self.w).reshape(-1)+self.b-y
        dw = 2*torch.mv(x.t(), tmp)
        db = 2*tmp
        return [dw, db]

    def train(self, x, y, lr=0.01, it=100, is_draw="True"):
        N, _ = x.shape
        loss_tmp = self.loss(x, y)
        los = [loss_tmp]
        print('times: 0, loss: ', loss_tmp)
        dw, db = self.backword(x, y)
        for i in range(it):
            self.w -= lr*dw
            self.b -= lr*db
            loss_tmp = self.loss(x, y)
            los.append(loss_tmp)
            print('times: ', i+1, ', loss: ', loss_tmp)
            dw, db = self.backword(x, y)
        plt.plot(range(it+1), los)
        plt.show()
        self.b = self.b.sum()   # 这个我不知道
        if is_draw and self.w.shape[0]==1:
            plt.plot(x, y, 'ro')
            xx = torch.linspace(torch.min(x), torch.max(x))
            yy = xx*self.w+self.b
            plt.plot(xx, yy)
            plt.show()
        return [self.w, self.b]

    def predict(self, x):
        return torch.mv(x, self.w)

    def math_method(self, x, y):
        x = torch.cat((torch.ones(x.shape[0],1), x), dim=1)
        print(x)
        return torch.mm(torch.mm(torch.linalg.inv(torch.mm(x.t(), x)), x.t()), y.reshape(-1, 1))

D = torch.tensor([  [0.2, 0.7, 1], 
                    [0.3, 0.3, 1], 
                    [0.4, 0.5, 1], 
                    [0.6, 0.5, 1], 
                    [0.1, 0.4, 1], 
                    [0.4, 0.6, -1], 
                    [0.6, 0.2, -1], 
                    [0.7, 0.4, -1],
                    [0.8, 0.6, -1],
                    [0.7, 0.5, -1]])
x = D[:, 0].reshape(-1, 1)
y = D[:, 1]

model = LinearRegression(x.shape[0], 1)
print(model.train(x, y))
print(model.math_method(x, y))
