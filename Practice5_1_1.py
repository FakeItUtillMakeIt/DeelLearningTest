from mxnet import autograd,nd
from mxnet.gluon import nn

def corr2d(x,y):
    z=nd.zeros(shape=(x.shape[0]-y.shape[0]+1,x.shape[1]-y.shape[1]+1))
    for i in range(x.shape[0]-y.shape[0]+1):
        for j in range(x.shape[1]-y.shape[1]+1):
            z[i,j]=(x[i:i+y.shape[0],j:j+y.shape[1]]*y).sum()
    return z

x=nd.ones(shape=(6,8))
x[:,2:6]=0
y=nd.array([[1,-1]])

z=corr2d(x,y)

class Conv2D(nn.Block):
    def __init__(self,kernel_size,**kwargs):
        super(Conv2D,self).__init__(**kwargs)
        self.weight=self.params.get('weight',shape=kernel_size)
        self.bias=self.params.get('bias',shape=(1,))

    def forward(self, k):
        #return corr2d(k,self.weight.data())+self.bias.data()
        data=k.reshape((1,1,)+k.shape)
        #data=k
        weight=self.weight.data()
        weight=weight.reshape((1,1,)+weight.shape)
        bias=self.bias.data()
        kernel=self.weight.shape
        return nd.Convolution(data,weight,bias,kernel,num_filter=1)


conv2d=Conv2D((1,2))
conv2d.initialize()
#x=x.reshape((1,1,6,8))
#z=z.reshape((1,1,6,7))

for i in range(10):
    with autograd.record():
        z_hat=conv2d(x)
        lo=(z_hat-z)**2
    lo.backward()
    conv2d.weight.data()[:]-=3e-2*conv2d.weight.grad()
    print(lo.sum().asscalar())
