import torch
import matplotlib.pyplot as plt

# x(N, d)   N个样本, d维
class Pocket():
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

    def result(self, update_time=10, is_print="True", is_draw="True"):
        rate = self.update(update_time, is_print)
        if is_draw=="True":
            self.draw2D('train')
        if is_print=="True":
            print("train: correct rate: ", float(rate), ", w: ", self.w)
        if self.test_x == None:
            return [self.w, rate]
        else:
            test_rate = self.test(is_print, is_draw)
            return [self.w, rate, test_rate]
    
    def update(self, update_time=10, is_print="True"):
        N, _ = self.x.shape
        it = 0
        my_teg = torch.sign(torch.mv(self.x, self.w))
        min_num = sum((my_teg!=self.tag).int())
        err_num = min_num
        w = self.w
        while True:
            if it>update_time or err_num==0: 
                break
            choose_one = torch.argmax((my_teg<=0)*torch.rand(N))   # 随机选出一个错误分类的
            w += self.tag[choose_one]*self.x[choose_one]
            tmp_teg = torch.sign(torch.mv(self.x, w))
            err_num = sum((tmp_teg!=self.tag).int())
            if is_print=="True":
                print("times: ", it, ", correct rate: ", float(1-err_num/N))
            it+=1
            if err_num<min_num:
                min_num = err_num
                self.w = w
                if is_print=="True":
                    print("new w: ", w)
        return 1-min_num/N

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
        max_x, min_x= max(x), min(x)
        pos_x, pos_y = x[tag==1], y[tag==1]
        neg_x, neg_y = x[tag==-1], y[tag==-1]
        plt.plot(pos_x, pos_y, pos)
        plt.plot(neg_x, neg_y, neg)
        k = -(self.w[1]/self.w[2])
        b = -(self.w[0]/self.w[2])
        xx = torch.linspace(min_x, max_x)
        yy = k*xx+b
        plt.plot(xx, yy, '-')
        if self.test_x==None or mode=='test':
            plt.show()
            
    def test(self, is_print="True", is_draw="True"):
        N, _ = self.test_x.shape
        tag = torch.sign(torch.mv(self.test_x, self.w))
        if is_draw=="True":
            self.draw2D('test', tag)
        err = sum((tag!=self.test_tag).int())
        rate = 1-err/N
        if is_print=="True":
            print("test: correct rate: ", float(rate))
        return rate