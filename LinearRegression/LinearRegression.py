#!/usr/bin/env python
# coding=utf-8
import torch
import matplotlib.pyplot as plt


# 想不到怎么写比较好




# x(N, d)
# y(N, )
# w(d, )
# b(N, )
class LinearRegression():
    def __init__(self, N, in_dim):
        self.w = torch.rand(in_dim+1)*1e-5

    def loss(self, x, y):
        return ((torch.mv(x, self.w).reshape(-1,1)-y.reshape(-1, 1))**2).sum()

    def backword(self, x, y):
        tmp = torch.mv(x, self.w).reshape(-1)-y
        dw = 2*torch.mv(x.t(), tmp)
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
                dw= self.backword(x, y)
                self.w -= lr*dw
                los.append(self.loss(x, y))
        elif method=='SGD':
            for i in range(it):
                idx = torch.randperm(N)
                for j in range(N//batch_size+1):
                    batch_x = x[idx[j*batch_size:min(N, (j+1)*batch_size)]]
                    batch_y = y[idx[j*batch_size:min(N, (j+1)*batch_size)]]
                    dw= self.backword(batch_x, batch_y)
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
                        dw = self.backword(batch_x, batch_y)
                        v1 = beta1*v1+dw
                        self.w -= lr*v1
                    los.append(self.loss(x, y))
            elif method=='Ada':
                for i in range(it):
                    idx = torch.randperm(N)
                    for j in range(N//batch_size+1):
                        batch_x = x[idx[j*batch_size:min(N, (j+1)*batch_size)]]
                        batch_y = y[idx[j*batch_size:min(N, (j+1)*batch_size)]]
                        dw= self.backword(batch_x, batch_y)
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
                        dw = self.backword(batch_x, batch_y)
                        v1 = beta1*dw+(1-beta1)*dw**2
                        self.w -= lr*dw/torch.sqrt(v1.abs()+1e-7)
                    los.append(self.loss(x, y))
            elif method=='Adam':
                for i in range(it):
                    idx = torch.randperm(N)
                    for j in range(N//batch_size+1):
                        batch_x = x[idx[j*batch_size:min(N, (j+1)*batch_size)]]
                        batch_y = y[idx[j*batch_size:min(N, (j+1)*batch_size)]]
                        dw = self.backword(batch_x, batch_y)
                        v1 = v1*beta1+(1-beta1)*dw
                        v2 = beta2*v2+(1-beta2)*dw**2
                        self.w -= lr*v1/torch.sqrt(v2+1e-7)
                    los.append(self.loss(x, y))
        return los

    def train(self, x, y, lr=0.01, method='common', beta1=0.9, beta2=0.999, it=100, batch_size=100, is_draw="True"):
        N, _ = x.shape
        x = torch.cat((torch.ones(N).reshape(-1, 1), x), dim=1)
        # print('times: 0, loss: ', loss_tmp)
        los = self.update_grad(x, y, lr, method, beta1, beta2, it, batch_size)
        print("loss: ", los)
        plt.plot(range(len(los)), los)
        plt.show()
        if is_draw and self.w.shape[0]==2:
            plt.plot(x[:,1], y, 'ro')
            xx = torch.linspace(torch.min(x[:,1]), torch.max(x[:,1]))
            yy = xx*self.w[1]+self.w[0]
            plt.plot(xx, yy)
            plt.show()
        return self.w

    # 只有做分类时用到这个
    def classify_draw(self, pos_x, neg_x, test_pos_x=None, test_neg_x=None):
        if pos_x.shape[1]==2:
            plt.plot(pos_x[:, 0], pos_x[:, 1], 'bo')
            plt.plot(neg_x[:, 0], neg_x[:, 1], 'rx')
            if test_pos_x!=None:
                plt.plot(test_pos_x[:, 0], test_pos_x[:, 1], 'c^')
                plt.plot(test_neg_x[:, 0], test_neg_x[:, 1], 'y*')
            min_x = torch.min(torch.min(pos_x[:, 0]), torch.min(neg_x[:, 0])).item()
            max_x = torch.max(torch.min(pos_x[:, 0]), torch.max(neg_x[:, 0])).item()
            xx = torch.linspace(min_x, max_x)
            w0, w1, w2 = self.w
            yy = -(w1*xx+w0)/w2
            plt.plot(xx, yy)
            plt.show()
        correct_num = sum(torch.sign(self.predict(pos_x))==1)+sum(torch.sign(self.predict(neg_x))==-1)
        rate = correct_num/(pos_x.shape[0]+neg_x.shape[0])
        print("train: correct rate: ", rate)
        if test_pos_x!=None:
            test_correct_num = sum(torch.sign(self.predict(test_pos_x))==1)+sum(torch.sign(self.predict(test_neg_x))==-1)
            test_rate = test_correct_num/(test_pos_x.shape[0]+test_neg_x.shape[0])
            print("test: correct rate: ", test_rate)
            return [rate, test_rate]
        return rate

    def predict(self, x):
        N = x.shape[0]
        x = torch.cat((torch.ones(N).reshape(-1, 1), x), dim=1)
        return torch.mv(x, self.w)

    def math_method(self, x, y):
        x = torch.cat((torch.ones(x.shape[0],1).reshape(-1,1), x), dim=1)
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
x = D[:, :2]
y = D[:, 2]

model = LinearRegression(x.shape[0], x.shape[1])
print(model.train(x, y, method='common'))
print(model.math_method(x, y))
model.classify_draw(D[:5, :2], D[5:, :2])
model.predict(x)