from mxnet import nd,autograd
from mxnet.gluon import nn

conv2d=nn.Conv2D(1,kernel_size=(1,2))
conv2d.initialize()

def corr2d(x,k):
    h,w=k.shape
    y=nd.zeros((x.shape[0]-h+1,x.shape[1]-w+1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,j]=(x[i:i+h,j:j+w]*k).sum()
    return y

x=nd.ones(shape=(6,8))
x[:,2:6]=0
z=nd.array([[1,-1]])

y=corr2d(x,z)

x=x.reshape((1,1,6,8))
y=y.reshape((1,1,6,7))

for i in range(10):
    with autograd.record():
        y_hat=conv2d(x)
        lo=(y_hat-y)**2
    lo.backward()
    conv2d.weight.data()[:]-=3e-2*conv2d.weight.grad()
    print(conv2d.weight.data().reshape(1,2))
    if (i+1)%2==0:
        print('batch%d,loss%.3f'%(i+1,lo.sum().asscalar()))