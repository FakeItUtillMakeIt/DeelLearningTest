#多通道输入和多通道输出
#通道即为channel，如彩色图像在h*w之外还有rgb三个颜色通道

import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import nn


#输入通道数为多通道，输出通道数为1输出通道
def corr2d(X,K):
    #首先沿着X和K的第0维（通道维）遍历，然后使用*将结果列表变成add_n
    #函数的位置参数来进行相加
    #d2l.corr2d(x,k)作用是矩阵对应元素相乘
    return nd.add_n(*[d2l.corr2d(x,k) for x,k in zip(X,K)])


x=nd.random.randint(low=1,high=4,shape=(2,3,3))
k=nd.random.randint(low=1,high=4,shape=(2,2,2))

z=corr2d(x,k)

# conv3d=nn.Conv3D(2,kernel_size=(2,2,2))
# x=x.reshape((2,1,)+x.shape[1:])
# conv3d.initialize()
# z1=conv3d(x)


#输出通道数和输入通道数等同
def corr2d_mutli_in_out(X,K):
    #对K的第0维遍历，每次同输入X做互相关计算，所有结果使用stack函数合并在一起
    return nd.stack(*[corr2d(X,k) for k in K])
K=k
#将每个通道的卷积核变为卷积核组【
K=nd.stack(K,K+1,K+2)

z1=corr2d_mutli_in_out(x,K)

#使用全连接层的矩阵乘法实现1*1卷积层
def corr2d_mutli(X,K):
    c_i,w,h=X.shape
    c_o,c_b=K.shape[:2]
    X=X.reshape((c_i,w*h))
    K=K.reshape((c_o,c_i))
    return nd.dot(K,X).reshape(c_o,h,w)

X=nd.random.uniform(shape=(3,3,3))
Y=nd.random.uniform(shape=(2,3,1,1))

Z1=corr2d_mutli(X,Y)
Z2=corr2d_mutli_in_out(X,Y)