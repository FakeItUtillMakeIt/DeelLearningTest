#卷积神经网络

from mxnet import autograd,nd
from mxnet.gluon import nn

#卷积操作
def corr2d(x,k):
    h,w=k.shape
    y=nd.zeros((x.shape[0]-h+1,x.shape[1]-w+1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,j]=(x[i:i+h,j:j+w]*k).sum()
    return y

#x=nd.array([[0,1,2],[3,4,5],[6,7,8]])
#y=nd.array([[0,1],[2,3]])

x=nd.ones(shape=(6,8))
x[:,2:6]=0
y=nd.array([[1,-1]])


z=corr2d(x,y)
print(z)