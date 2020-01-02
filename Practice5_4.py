#池化层

from mxnet import nd
from mxnet.gluon import nn

#池化层操作
def pool2d(X,size,mode='max'):
    p_h,p_w=size
    Y=nd.zeros(shape=(X.shape[0]-p_h+1,X.shape[1]-p_w+1))
    for i in range(X.shape[0]-p_h+1):
        for j in range(X.shape[1]-p_w+1):
            if mode=='max':
                Y[i,j]=X[i:i+p_h,j:j+p_w].max()
            elif mode=='svg':
                Y[i,j]=X[i:i+p_h,j:j+p_w].mean()
    return Y

x=nd.random.randint(low=0,high=10,shape=(3,3))
pool_size=(2,2)

z=pool2d(x,pool_size,mode='max')

#填充和步幅
X=nd.arange(16).reshape(shape=(1,1,4,4))

#调用nn库的最大池化函数
pool2d1=nn.MaxPool2D(pool_size=(3,3))
pool2d1(X)

pool2d2=nn.MaxPool2D(pool_size=(3,3),strides=1,padding=1)

#改变输入通道数
Y=nd.concat(X,X+1,dim=1)#dim=1按列加


