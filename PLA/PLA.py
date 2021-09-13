import torch
import matplotlib.pyplot as plt

# x(N, d)   N个样本, d维
class PLA():
    def __init__(self, x, tag, test_x=None, test_tag=None):
        N, d = x.shape
        self.x = torch.cat((torch.ones(N).reshape(N, 1), x), dim=1)
        self.tag = tag
        self.w = torch.zeros(d+1)
        self.test_x = test_x
        self.test_tag = test_tag
        if test_x!=None:
            M, _ = test_x.shape
            self.test_x = torch.cat((torch.ones(M).reshape(M, 1), test_x), dim=1)

    def result(self, is_print="True", is_draw="True"):
        rate = self.update(is_print)
        if is_draw:
            self.draw2D('train')
        if is_print:
            print("train: correct rate: ", float(rate), ", w: ", self.w)
        if self.test_x == None:
            return [self.w, rate]
        else:
            test_rate = self.test(is_print, is_draw)
            return [self.w, rate, test_rate]
    
    def update(self, is_print="True"):
        N, _ = self.x.shape
        it = 0
        idx = -1
        err = N
        while err:
            my_tag = torch.sign(torch.mv(self.x, self.w))
            err = sum((my_tag!=self.tag).int())
            if is_print:
                print("times: ", it, ", correct rate: ", float(1-err/N), ", w", self.w)
            for i in range(N):
                idx = (idx+1)%N
                if(my_tag[idx]!=self.tag[idx]):
                    it+=1
                    self.w += self.tag[idx]*self.x[idx]
                    break
        return 1-err/N

    def draw2D(self, mode='train', test_my_tag=None):
        assert(self.w.shape[0]==3)
        if mode == 'train':
            x, y = self.x[:, 1], self.x[:, 2]
            tag = self.tag
            pos = 'bo'
            neg = 'rx'
        elif mode == 'test':
            x, y = self.test_x[:, 1], self.test_x[:, 2]
            tag = test_my_tag
            pos = 'cp'
            neg = 'y^'
        max_x, max_y, min_x, min_y = max(x), max(y), min(x), min(y)
        pos_x, pos_y = x[tag==1], y[tag==1]
        neg_x, neg_y = x[tag==-1], y[tag==-1]
        plt.plot(pos_x, pos_y, pos)
        plt.plot(neg_x, neg_y, neg)
        k = -(self.w[1]/self.w[2])
        b = -(self.w[0]/self.w[2])
        xx = torch.linspace(min_x, max_x)
        yy = k*xx+b
        plt.plot(xx, yy, '-')
        # flag = max(yy)>max_y and min(yy)<min_x
        # min_yy = min_y if flag else min(yy)
        # max_yy = max_y if flag else max(yy)
        # plt.axis([min_x, max_x, min(min_yy, min_y), max(max_yy, max_y)])
        if self.test_x==None or mode=='test':
            plt.show()

    def test(self, is_print="True", is_draw="True"):
        N, _ = self.test_x.shape
        tag = torch.sign(torch.mv(self.test_x, self.w))
        if is_draw:
            self.draw2D('test', tag)
        err = sum((tag!=self.test_tag).int())
        rate = 1-err/N
        if is_print:
            print("test: correct rate: ", float(rate))
        return rate
