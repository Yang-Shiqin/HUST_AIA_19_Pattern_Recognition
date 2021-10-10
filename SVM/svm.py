#!/usr/bin/env python
# coding=utf-8
import torch
import cvxopt
from cvxopt import matrix, solvers      # ATTENTION: matrix是转置的样子
import random
import matplotlib.pyplot as plt

# x(N, d)
# y(N, 1)

class SVM_base:
    def __init__(self):
        self.alpha = None

    def QP(self, x, y):         # 求解二次规划
        raise NotImplementedError 

    def train(self, x, y):
        raise NotImplementedError 

    def plot(self, x, y, test_x=None, test_y=None):
        assert(x.shape[1]==2)
        x = x.float()
        y = y.reshape(-1).float()
        pos_x = x[y==1]
        neg_x = x[y<=0]
        if test_x!=None:
            pos_test = test_x[test_y==1]
            neg_test = test_x[test_y<=0]
            plt.plot(pos_test[:, 0], pos_test[:, 1], 'c^')
            plt.plot(neg_test[:, 0], neg_test[:, 1], 'gp')
        w1, w2=self.w
        w0 = self.b
        xx = torch.linspace(torch.min(x[:, 0]), torch.max(x[:, 0]))
        yy = -w0/w2-w1/w2*xx
        yy1 = -(w0+1)/w2-w1/w2*xx       # neg
        yy2 = -(w0-1)/w2-w1/w2*xx       # pos
        plt.plot(xx, yy)
        plt.plot(xx, yy1, '--')
        plt.plot(xx, yy2, '--')
        if self.alpha==None:
            p_support = pos_x[abs((pos_x@self.w+self.b).reshape(-1)-1)<=1e-5]     # 画出支持向量
            n_support = neg_x[abs((neg_x@self.w+self.b).reshape(-1)+1)<=1e-5]
            print("pos supporter: \n", p_support)
            print("neg supporter: \n", n_support)
            plt.plot(p_support[:, 0], p_support[:, 1], 'yo', markersize=15)
            plt.plot(n_support[:, 0], n_support[:, 1], 'yo', markersize=15)
        else:
            support = x[(self.alpha).reshape(-1)>=1e-5]
            print("supporter: \n", support)
            plt.plot(support[:, 0], support[:, 1], 'yo', markersize=15)
        plt.plot(pos_x[:, 0], pos_x[:, 1], 'bo')
        plt.plot(neg_x[:, 0], neg_x[:, 1], 'rx')
        plt.show()

    def predict(self, x, y, test_x=None, test_y=None):
        w = self.w.reshape(-1, 1)
        N = x.shape[0]
        N2 = test_x.shape[0]
        y = y.reshape(-1)
        train = (x@w+self.b).reshape(-1)>0
        rate = sum(((train)==(y==1)).int())/N
        print("train correct rate: ", rate)
        if test_x!=None:
            test_y = test_y.reshape(-1)
            test = (test_x@w+self.b).reshape(-1)>0
            test_rate = sum(((test)==(test_y==1)).int())/N2
            print("test correct rate: ", test_rate)
        

class Primal_SVM(SVM_base):
    def QP(self, x, y):
        N, d = x.shape
        x = torch.cat((torch.ones(N,1), x), dim=1)
        P = torch.eye(d+1)
        P[0][0] = 0
        P = matrix(P.double().numpy().tolist())    # 必须转换为float
        q = matrix([0.]*(d+1))
        G = matrix((-y*x).double().t().numpy().tolist())
        h = matrix([-1.]*N)
        A=None
        b=None
        result = solvers.qp(P,q,G,h,A,b)
        w = torch.tensor(list(result['x']))
        return w

    def train(self, x, y):
        y = y.reshape(-1, 1)
        w = self.QP(x, y).reshape(-1,1)
        self.b = w[0]
        self.w = w[1:]
        y = y.reshape(-1)
        return [self.b, self.w]



class Dual_SVM(SVM_base):
    def QP(self, x, y):
        N, _ = x.shape
        P = matrix((y.t()*(x@x.t())*y).double().t().numpy().tolist())    # 必须转换为double
        q = matrix([-1.]*N)
        G = matrix((-1*torch.eye(N)).double().numpy().tolist())
        h = matrix([0.]*N)
        A = matrix(y.double().numpy().tolist())
        b = matrix([0.])
        result = solvers.qp(P,q,G,h,A,b)
        alpha = torch.tensor(list(result['x']))
        self.alpha = alpha
        return alpha

    def train(self, x, y):
        x = x.float()
        y = y.reshape(-1, 1).float()
        alpha = self.QP(x, y).reshape(-1,1)
        self.w = ((alpha*y).t()@x).reshape(-1)
        y = y.reshape(-1)
        pos_x = x[y==1]
        neg_x = x[y<=0]
        y = y.reshape(-1,1)
        # self.b = -(torch.max(neg_x@self.w)+torch.min(pos_x@self.w))/2
        self.b = -(torch.max((neg_x@x.t()@(alpha*y)).t())+torch.min(pos_x@x.t()@(alpha*y)))/2   # 展开w, 方便kernel
        return [self.b, self.w]

class Kernel_SVM(SVM_base):
    # type == gauss -> Gaussian kernel
    # type is a int -> Polynomial kernel
    # '2'和'4'是固定参数多项式核(一般表达式拟合不了的常用参数)
    def K(self, x1, x2, method='gauss', beta1=1, beta2=1):
        assert((method in['gauss', '2', '4']) or (isinstance(method,int) and method>0))
        if method=='gauss':
            xx1 = (x1*x1).sum(dim=1)
            xx2 = (x2*x2).sum(dim=1)
            z = 2*(x1@x2.t())-xx1.reshape(-1,1)-xx2.reshape(1,-1)
            return torch.exp(beta1*z)
        elif method=='2':
            return 1+x1@x2.t()+(x1@x2.t())**2
        elif method=='4':
            return 1+x1@x2.t()+0.01*(x1@x2.t())**4
        else:
            return (beta1*(x1@x2.t())+beta2)**method

    
    def QP(self, x, y):
        N, _ = x.shape
        y = y.reshape(-1,1).double()
        x = x.double()
        P = matrix((y.t()*(x)*y).t().numpy().tolist())    # 必须转换为float
        q = matrix([-1.]*N)
        G = matrix((-1*torch.eye(N)).double().numpy().tolist())
        h = matrix([0.]*N)
        A = matrix(y.numpy().tolist())
        b = matrix([0.])
        result = solvers.qp(P,q,G,h,A,b)
        alpha = torch.tensor(list(result['x']))
        return alpha

    def train(self, x, y, method='gauss', beta1=1, beta2=1):
        self.method = method
        self.beta1 = beta1
        self.beta2 = beta2
        x = x.float()
        y = y.reshape(-1)
        pos_x = x[y==1]
        neg_x = x[y<=0]
        y = y.reshape(-1, 1).float()
        z = self.K(x, x, method, beta1, beta2)
        alpha = self.QP(z, y).reshape(-1,1)
        self.alpha = alpha
        self.b = -(torch.max((self.K(neg_x, x, method, beta1, beta2)@(alpha*y)).t())+\
            torch.min(self.K(pos_x, x, method, beta1, beta2)@(alpha*y)))/2
        return alpha

    def plot(self, x, y, test_x=None, test_y=None, n=21):
        assert(x.shape[1]==2)
        x = x.float()
        y = y.reshape(-1).float()
        pos_x = x[y==1]
        neg_x = x[y<=0]
        min_x = torch.min(x[:,0])
        max_x = torch.max(x[:,0])
        min_y = torch.min(x[:,1])
        max_y = torch.max(x[:,1])
        if test_x!=None:
            pos_test = test_x[test_y==1]
            neg_test = test_x[test_y<=0]
            plt.plot(pos_test[:, 0], pos_test[:, 1], 'g^')
            plt.plot(neg_test[:, 0], neg_test[:, 1], 'cp')
            min_x = min(min_x, torch.min(test_x[:,0]))
            max_x = max(max_x, torch.max(test_x[:,0]))
            min_y = min(min_y, torch.min(test_x[:,1]))
            max_y = max(max_y, torch.max(test_x[:,1]))

        xx = torch.linspace(min_x, max_x,n)
        yy = torch.linspace(min_y, max_y,n)
        X,Y = torch.meshgrid(xx, yy)
        xk = x[(self.alpha).reshape(-1)>=1e-5]
        yk = y[(self.alpha).reshape(-1)>=1e-5]
        self.support = self.alpha[self.alpha>=1e-5]
        xxx = torch.transpose(torch.transpose(torch.stack((X, Y),dim=0),0,2),1,0).reshape(-1,2)
        f = ((self.alpha.reshape(1,-1)*y.reshape(1,-1))@self.K(x, xxx, self.method, self.beta1, self.beta2)).reshape(n,n)+self.b
        contour = plt.contourf(X, Y, f,levels=10, alpha=.75, cmap='coolwarm')
        CS = plt.contour(X, Y, f, linewidths=1, linestyles='dashed', levels=[-1, 0, 1], colors='k')
        plt.clabel(CS, inline=True)
        plt.colorbar(contour)

        support = xk
        self.support_x = support
        self.support_y = yk
        print("supporter: \n", support)
        plt.plot(support[:, 0], support[:, 1], 'yo', markersize=10)
        plt.plot(pos_x[:, 0], pos_x[:, 1], 'ro', markersize=5)
        plt.plot(neg_x[:, 0], neg_x[:, 1], 'bx', markersize=5)
        plt.show()

    def predict(self, x, y, test_x=None, test_y=None):
        yk = self.support_y
        xk = self.support_x
        f = ((self.support.reshape(1,-1)*yk.reshape(1,-1))@self.K(xk, x, self.method, self.beta1, self.beta2))+self.b
        rate = sum(((f.squeeze()>0)==(y.squeeze()>0)).int())/x.shape[0]
        print("train: correct rate: ", rate)
        if test_x!=None:
            f = ((self.support.reshape(1,-1)*yk.reshape(1,-1))@self.K(xk, test_x, self.method, self.beta1, self.beta2))+self.b
            test_rate = sum(((f.squeeze()>0)==(test_y.squeeze()>0)).int())/test_x.shape[0]
            print("test: correct rate: ", test_rate)

################################## test ####################################
if __name__=='__main__':
    solvers.options['show_progress'] = False    # 抑制输出

    # 生成数据
    N = 200
    pos_x = torch.randn(N, 2)+torch.tensor([-5, 0])
    neg_x = torch.randn(N, 2)+torch.tensor([0, 5])
    x = torch.cat((pos_x, neg_x), dim=0)
    pos_y = torch.ones(N,)
    neg_y = -torch.ones(N,)
    y = torch.cat((pos_y, neg_y), dim=0)
    N, d = pos_x.shape
    idx = [i for i in range(2*N)]
    random.shuffle(idx)
    train_x, train_y = x[idx[:int(2*N*0.8)]], y[idx[:int(2*N*0.8)]]
    test_x, test_y = x[idx[int(2*N*0.8):]], y[idx[int(2*N*0.8):]]
    
    # # Primal_SVM
    # model = Primal_SVM()
    # b, w = model.train(train_x, train_y)
    # print("b: ", b, ", w: ", w.reshape(-1))
    # model.plot(train_x, train_y, test_x, test_y)
    # model.predict(train_x, train_y, test_x, test_y)

    # # Dual_SVM
    # model = Dual_SVM()
    # b, w = model.train(train_x, train_y)
    # print("b: ", b, ", w: ", w.reshape(-1))
    # model.plot(train_x, train_y, test_x, test_y)
    # model.predict(train_x, train_y, test_x, test_y)

    # Kernel_SVM
    model = Kernel_SVM()
    # model.train(train_x, train_y, method='2')       # 二次多项式核
    model.train(train_x, train_y, method='gauss', beta1=0.1)    # gauss核
    # model.train(train_x, train_y, method='4')       # 四次多项式核
    model.plot(train_x, train_y, test_x, test_y)
    model.predict(train_x, train_y, test_x, test_y)

################# test: 异或 ######################
    # x = torch.tensor([[2,2],
    #           [-2,-2],
    #           [2,-2],
    #           [-2,2]])
    # y = torch.tensor([1, 1,-1,-1])
    # model = Kernel_SVM()
    # w = model.train(x, y, method=2)
    # model.plot(x, y)
    # print(w)

