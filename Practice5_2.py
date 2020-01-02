from mxnet import nd
from mxnet.gluon import nn

def conp_conv2d(conv2d,x):
    conv2d.initialize()
    x=x.reshape((1,1)+x.shape)
    y=conv2d(x)
    return y.reshape(y.shape[2:])

# conv2d=nn.Conv2D(1,kernel_size=3,padding=1)
# x=nd.random.uniform(shape=(8,8))
# shape=conp_conv2d(conv2d,x).shape

# conv2d=nn.Conv2D(1,kernel_size=3,strides=1,padding=1)
# x=nd.random.uniform(shape=(6,6))
# shape=conp_conv2d(conv2d,x)

conv2d=nn.Conv2D(1,kernel_size=3,strides=2,padding=2)
x=nd.random.uniform(shape=(6,6))
shape=conp_conv2d(conv2d,x).shape

conv2d=nn.Conv2D(1,kernel_size=(3,5),strides=(3,2),padding=2)
shape=conp_conv2d(conv2d,x).shape
