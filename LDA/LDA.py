#!/usr/bin/env python
# coding=utf-8
import torch
import matplotlib.pyplot as plt
import numpy as np    # torch求逆有些数据结果很不准(但是转换后精度会下降, 误差还是很大, 暂时还没解决办法)


################################### LDA ######################################3

# 暂时只支持2分类
class LDA:
    def __init__(self, in_dim):
        self.w = torch.zeros(in_dim)
        self.s = 0          # 分类阈值
        self.mu = None

    def get_w(self, x1, x2):  # 求特征向量
        N1, N2 = x1.shape[0], x2.shape[0]
        mu1 = x1.mean(dim=0)
        mu2 = x2.mean(dim=0)
        Sb = torch.mm((mu1-mu2).reshape(-1, 1), (mu1-mu2).reshape(1, -1))
        Sw = torch.mm((x1-mu1).t(), x1-mu1)/N1+torch.mm((x2-mu2).t(), x2-mu2)/N2
        # print(Sb, Sw)
        # print("ss: ", torch.mm(torch.inverse(Sb), Sw))
        # print("ysq", Sb@torch.linalg.inv(Sb))
        self.w, _ = torch.linalg.eig(torch.mm(torch.inverse(Sb), Sw))    # 
        self.mu = (mu1+mu2)/2
        self.s = torch.dot(self.w.float(), self.mu)
        return [self.w, self.s]

    def get_w2(self, x1, x2): # 不用求特征向量
        N1, N2 = x1.shape[0], x2.shape[0]
        mu1 = x1.mean(dim=0)
        mu2 = x2.mean(dim=0)
        Sw = torch.mm((x1-mu1).t(), x1-mu1)/N1+torch.mm((x2-mu2).t(), x2-mu2)/N2
        self.w = (torch.inverse(Sw)@(mu1-mu2).reshape(-1,1)).reshape(-1)
        self.mu = (mu1+mu2)/2
        self.s = torch.dot(self.w.float(), self.mu)
        return [self.w, self.s]

    def predict(self, x):
        return torch.sign(-torch.mv(x, self.w.float())+self.s)

    def draw(self, x1, x2, test_x1=None, test_x2=None):
        assert(x1.shape[1]==2)
        plt.plot(x1[:, 0], x1[:, 1], 'bo')
        plt.plot(x2[:, 0], x2[:, 1], 'rx')
        if test_x1!=None:
            plt.plot(test_x1[:, 0], test_x1[:, 1], 'c^')
            plt.plot(test_x2[:, 0], test_x2[:, 1], 'yp')
        minx = torch.min(torch.min(x1[:, 0]), torch.min(x2[:, 0]))
        maxx = torch.max(torch.max(x1[:, 0]), torch.max(x2[:, 0]))
        miny = torch.min(torch.min(x1[:, 1]), torch.min(x2[:, 1]))
        maxy = torch.max(torch.max(x1[:, 1]), torch.max(x2[:, 1]))
        midx = (minx+maxx)/2
        midy = (miny+maxy)/2
        xx = torch.linspace(minx, maxx)
        w1, w2 = self.w
        
        yy = -w1/w2*(xx-midx)+midy
        # if w1.float()<1e-7:
        #   plt.vlines(self.mu[0], miny, maxy)
        # else:
        ss = w2/w1*(xx-self.mu[0])+self.mu[1]
        plt.plot(xx, ss, '--')
        # print(w2, w1)
        # print(ss)
        plt.plot(xx, yy)
        # plt.ylim((miny, maxy))
        plt.show()


################################### test ######################################3

pos_x = torch.randn(200, 2)+torch.tensor([-5, 0])
neg_x = torch.randn(200, 2)+torch.tensor([0, 5])
x = torch.cat((pos_x, neg_x), dim=0)
N, d = pos_x.shape
train_pos_x = pos_x[:int(N*0.8)]
train_neg_x = neg_x[:int(N*0.8)]
test_pos_x = pos_x[int(N*0.8):]
test_neg_x = neg_x[int(N*0.8):]

model = LDA(x.shape[1])
w, s = model.get_w2(train_pos_x, train_neg_x)
rate1 = (sum(model.predict(train_pos_x)==1)+sum(model.predict(train_neg_x)==-1))/train_pos_x.shape[0]/2
rate1 = max(rate1, 1-rate1)
rate2 = (sum(model.predict(test_pos_x)==1)+sum(model.predict(test_neg_x)==-1))/test_pos_x.shape[0]/2
rate2 = max(rate2, 1-rate2)
print("train: correct rate: ", rate1)
print("test: correct rate: ", rate2)
model.draw(train_pos_x, train_neg_x, test_pos_x, test_neg_x)
print("threshold: ", s)
