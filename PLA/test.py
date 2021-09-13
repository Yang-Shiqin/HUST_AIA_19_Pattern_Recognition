from PLA import PLA
from Pocket import Pocket
import torch
import time

# test1
x = torch.tensor([  [0.2, 0.7], 
                    [0.3, 0.3], 
                    [0.4, 0.5], 
                    [0.6, 0.5], 
                    [0.1, 0.4], 
                    [0.4, 0.6],
                    [0.6, 0.2],
                    [0.7, 0.4], 
                    [0.8, 0.6], 
                    [0.7, 0.5]])
tag = torch.tensor([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
model = Pocket(x, tag)
model.result(20)


# test2
import time
x1 = torch.randn(200, 2)+torch.tensor([-5, 0])
x2 = torch.randn(200, 2)+torch.tensor([0, -5])
train_num = int(200*0.8)
test_num = 200-train_num
x1_train = x1[:train_num]
x1_test = x1[train_num:]
x2_train = x2[:train_num]
x2_test = x2[train_num:]
x = torch.cat((x1_train, x2_train), dim=0)
tag = torch.cat((torch.ones(train_num), -torch.ones(train_num)), dim=0)
test_x = torch.cat((x1_test, x2_test), dim=0)
test_tag = torch.cat((torch.ones(test_num), -torch.ones(test_num)), dim=0)

start = time.clock()
pla = PLA(x, tag, test_x, test_tag)
w1, _, _ = pla.result()
end = time.clock()
print('PLA run_time: ', end-start)

print('='*40)
start = time.clock()
pocket = Pocket(x, tag, test_x, test_tag)
w2, _, _ = pocket.result(20)
end = time.clock()
print('Pocket run_time: ', end-start)
