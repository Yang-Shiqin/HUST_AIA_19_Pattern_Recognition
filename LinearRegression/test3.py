import torch
import random
import math
# from LinearRegression import LinearRegression
def degrad(x, dx, method='common', config=None, beta1=0.9, beta2=0.999, lr=0.4):
    if method=='common':
        x -= lr*dx
    elif method=='SGD':
        x -= lr*torch.rand(1).item()*dx
    elif method=='M': 
        v = config['v']
        v = beta1*v+dx
        x -= lr*v
        config['v'] = v
    elif method=='Ada':
        dx2 = config['dx2']
        dx2 += dx**2
        x -= lr*dx/torch.sqrt(dx2+1e-6)
        config['dx2'] = dx2
    elif method=='RMS':       # NAN(梯度为-, 不能开方), 加abs
        v = beta1*dx+(1-beta1)*dx**2
        x -= lr*dx/torch.sqrt(v.abs()+1e-6)
        
    elif method=='Adam':
        v1 = config['v1']
        v2 = config['v2']
        v1 = v1*beta1+(1-beta1)*dx
        v2 = beta2*v2+(1-beta2)*dx**2
        x -= lr*v1/torch.sqrt(v2+1e-6)
        config['v1'] = v1
        config['v2'] = v2
    config['x'] = x
    return config
    
# test3 f(x)=x*cos(0.25π*x)
pi = math.pi
it = 20
x = torch.tensor(-4.)
method = 'Adam'
if method in ['common', 'SGD', 'RMS']:
  config = {'x': x}
elif method == 'M':
  config = {'x': x, 'v': torch.tensor(0.)}
elif method == 'Ada':
  config = {'x': x, 'dx2': torch.tensor(0.)}
elif method=='Adam':
  config = {'x': x, 'v1': torch.tensor(0.), 'v2':torch.tensor(0.)}
xx = torch.linspace(-10, 10)
yy = xx*torch.cos(pi/4*xx)
# yyy = torch.cos(pi/4*xx)-pi/4*xx*torch.sin(pi/4*xx)
plt.plot(xx, yy)
# plt.plot(xx, yyy)
fx = x*torch.cos(pi/4*x)
for i in range(it):
    plt.plot(x, fx, 'ro')
    dx = torch.cos(pi/4*x)-pi/4*x*torch.sin(pi/4*x)
    print(dx, x)
    config = degrad(x, dx, method=method, config=config)
    x = config['x']
    fx = x*torch.cos(pi/4*x)