import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import DataLoader, sampler

# def load_mnist(path='./mnist', num_train=49000, num_val=1000, batch_size=64):
def load_mnist(path='./mnist', batch_size=64):
    transform = T.Compose([
                  T.ToTensor(),
                  T.Normalize((0.5), (0.5)) # mnist默认灰度图像, 就1个
              ])
    
    mnist_train = dset.MNIST(path, train=True, download=True, transform=transform)
    # loader_train = DataLoader(mnist_train, batch_size, sampler=sampler.SubsetRandomSampler(range(num_train)))
    loader_train = DataLoader(mnist_train, batch_size, shuffle = True)
    
    # mnist_val = dset.MNIST(path, train=True, download=True, transform=transform)
    # loader_val = DataLoader(mnist_val, batch_size, sampler=sampler.SubsetRandomSampler(range(num_train, num_train+num_val)))
    
    mnist_test = dset.MNIST(path, train=False, download=True, transform=transform)
    loader_test = DataLoader(mnist_test, batch_size, shuffle = True)
    
    # return loader_train, loader_val, loader_test
    return loader_train, loader_test

def acc(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on train set')
    else:
        print('Checking accuracy on test set')   
    correct = 0
    num = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            y=y.long()
            scores = model(x)
            _, pred_y = scores.max(1)
            correct += (pred_y==y).sum()
            num += pred_y.size(0)
        rate = correct.float()/num
        print('Got %d / %d correct (%.2f)' % (correct, num, 100 * rate))
    return rate

def train(model, loader_train, loader_test, optimizer, epoch=1):
    num_iters = epoch
    acc_history = torch.zeros(num_iters)
    acc_test_history = torch.zeros(num_iters)
    loss_history = torch.zeros(num_iters)
    iter_history = torch.zeros(num_iters)
    for e in range(epoch):
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.float()
            y = y.long()
            score = model(x)
            loss = F.cross_entropy(score, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tt = e*len(loader_train)+t

            if t==len(loader_train)-1:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, tt, loss.item()))
                acc_history[e] = acc(loader_train, model)
                acc_test_history[e] = acc(loader_test, model)
                loss_history[e] = loss.item()
                iter_history[e] = tt

    return acc_history, acc_test_history, loss_history, iter_history