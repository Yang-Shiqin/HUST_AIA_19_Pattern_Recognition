import torch
import random
from LinearRegression import LinearRegression

pos_x = torch.randn(200, 2)+torch.tensor([-5, 0])
neg_x = torch.randn(200, 2)+torch.tensor([0, 5])
x = torch.cat((pos_x, neg_x), dim=0)
pos_y = torch.ones(200,)
neg_y = -torch.ones(200,)
y = torch.cat((pos_y, neg_y), dim=0)
N, d = pos_x.shape
idx = [i for i in range(2*N)]
random.shuffle(idx)
train_x, train_y = x[idx[:int(2*N*0.8)]], y[idx[:int(2*N*0.8)]]
test_x, test_y = x[idx[int(2*N*0.8):]], y[idx[int(2*N*0.8):]]
model = LinearRegression(int(2*N*0.8), d)
print(model.train(train_x, train_y, method='Ada', it=2000, lr = 0.01))    # w

model.classify_draw(train_x[train_y==1], train_x[train_y==-1], test_x[test_y==1], test_x[test_y==-1])