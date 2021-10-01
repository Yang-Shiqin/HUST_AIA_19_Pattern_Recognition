#!/usr/bin/env python
# coding=utf-8
import torch
import matplotlib.pyplot as plt
import numpy as np

class LogisticRegression:
    def __init__(self, dim_in):
        self.w = torch.rand(dim_in+1)

    def predict(self, x):
        N, d = x.shape
        x = torch.cat((torch.ones(N,1), x), dim=1)
        return torch.sigmoid(torch.mv(x, self.w))

    def loss(self, x, y):
        hx = self.predict(x)
        # 交叉熵求loss的两种改进(防止nan)
        # los = (y*torch.log(hx+1e-10)).sum()+((1-y)*torch.log(1-hx+1e-10)).sum()
        los = -(y*torch.tensor(np.nan_to_num(torch.log(hx)))).sum()-((1-y)*torch.tensor(np.nan_to_num(torch.log(1-hx)))).sum()
        return los

    def backward(self, x, y):
        N, _ = x.shape
        sig = self.predict(x).reshape(-1, 1)
        x = torch.cat((torch.ones(N,1), x), dim=1)
        dw = x.t()@((1-sig)*y.reshape(-1, 1))+x.t()@(-sig*(1-y.reshape(-1, 1)))
        dw = -dw.reshape(-1)
        return dw

    # common    : 梯度下降法
    # SGD       : 随机梯度下降法    beta1-采样比例
    # M         : 动量法            beta1
    # Ada       : Adagrad           beta1
    # RMS       : RMSProp           beta1 beta2
    # Adam      : Adam              beta1 beta2
    def update_grad(self, x, y, lr, method='common', beta1=0.9, beta2=0.999, it=100, batch_size=128):
        N, d = x.shape
        
        los = []
        if method=='common':
            for i in range(it):
                dw= self.backward(x, y)
                self.w -= lr*dw
                los.append(self.loss(x, y))
        elif method=='SGD':
            for i in range(it):
                idx = torch.randperm(N)
                for j in range(N//batch_size+1):
                    batch_x = x[idx[j*batch_size:min(N, (j+1)*batch_size)]]
                    batch_y = y[idx[j*batch_size:min(N, (j+1)*batch_size)]]
                    dw= self.backward(batch_x, batch_y)
                    self.w -= lr*dw
                los.append(self.loss(x, y))
        else:
            v1 = 0
            v2 = 0
            if method=='M': 
                for i in range(it):
                    idx = torch.randperm(N)
                    for j in range(N//batch_size+1):
                        batch_x = x[idx[j*batch_size:min(N, (j+1)*batch_size)]]
                        batch_y = y[idx[j*batch_size:min(N, (j+1)*batch_size)]]
                        dw = self.backward(batch_x, batch_y)
                        v1 = beta1*v1+dw
                        self.w -= lr*v1
                    los.append(self.loss(x, y))
            elif method=='Ada':
                for i in range(it):
                    idx = torch.randperm(N)
                    for j in range(N//batch_size+1):
                        batch_x = x[idx[j*batch_size:min(N, (j+1)*batch_size)]]
                        batch_y = y[idx[j*batch_size:min(N, (j+1)*batch_size)]]
                        dw= self.backward(batch_x, batch_y)
                        v1 = dw
                        v2 += v1**2
                        self.w -= lr*v1/torch.sqrt(v2+1e-7)
                    los.append(self.loss(x, y))
            elif method=='RMS':       # NAN(梯度为-, 不能开方), 加abs
                for i in range(it):
                    idx = torch.randperm(N)
                    for j in range(N//batch_size+1):
                        batch_x = x[idx[j*batch_size:min(N, (j+1)*batch_size)]]
                        batch_y = y[idx[j*batch_size:min(N, (j+1)*batch_size)]]
                        dw = self.backward(batch_x, batch_y)
                        v1 = beta1*dw+(1-beta1)*dw**2
                        self.w -= lr*dw/torch.sqrt(v1.abs()+1e-7)
                    los.append(self.loss(x, y))
            elif method=='Adam':
                for i in range(it):
                    idx = torch.randperm(N)
                    for j in range(N//batch_size+1):
                        batch_x = x[idx[j*batch_size:min(N, (j+1)*batch_size)]]
                        batch_y = y[idx[j*batch_size:min(N, (j+1)*batch_size)]]
                        dw = self.backward(batch_x, batch_y)
                        v1 = v1*beta1+(1-beta1)*dw
                        v2 = beta2*v2+(1-beta2)*dw**2
                        self.w -= lr*v1/torch.sqrt(v2+1e-7)
                    los.append(self.loss(x, y))
        return los

    def train(self, x, y, lr=0.01, method='common', beta1=0.9, beta2=0.999, it=100, batch_size=100, is_draw="True"):
        y[y<=0]=0
        los = self.update_grad(x, y, lr, method, beta1, beta2, it, batch_size)
        print("loss: ", los)
        plt.plot(range(len(los)), los)
        plt.show()
        # if is_draw and self.w.shape[0]==2:
        #     plt.plot(x[:,1], y, 'ro')
        #     xx = torch.linspace(torch.min(x[:,1]), torch.max(x[:,1]))
        #     yy = xx*self.w[1]+self.w[0]
        #     plt.plot(xx, yy)
        #     plt.show()
        return self.w

    def draw(self, x, y, test_x=None, test_y=None):
        assert(x.shape[1]==2)
        pos_x = x[y==1]
        neg_x = x[y<=0]
        plt.plot(pos_x[:, 0], pos_x[:, 1], 'bo')
        plt.plot(neg_x[:, 0], neg_x[:, 1], 'rx')
        train_rate = sum(((self.predict(x)>0.5)==(y==1)).int())/x.shape[0]
        print("train: correct rate: ", train_rate)
        if test_x!=None:
            pos_test = test_x[test_y==1]
            neg_test = test_x[test_y<=0]
            plt.plot(pos_test[:, 0], pos_test[:, 1], 'c^')
            plt.plot(neg_test[:, 0], neg_test[:, 1], 'yp')
            test_rate = sum(((self.predict(test_x)>0.5)==(test_y==1)).int())/test_x.shape[0]
            print("test: correct rate: ", test_rate)
        w0, w1, w2=self.w
        xx = torch.linspace(torch.min(x[:, 0]), torch.max(x[:, 0]))
        yy = -w0/w2-w1/w2*xx
        plt.plot(xx, yy)
        plt.show()
        


