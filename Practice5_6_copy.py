#AlexNet深度 卷积神经网络

import d2lzh as d2l
from mxnet import gluon,init,nd
from mxnet.gluon import data as gdata,nn
import os
import sys

# import tensorflow as tf
#
# hello=tf.constant('nihao')
# session=tf.Session()
# print(session.run(hello))

#AlexNet网络层布置
net=nn.Sequential()
net.add(#第一层
        nn.Conv2D(96,kernel_size=11,strides=4,activation='relu'),
        nn.MaxPool2D(pool_size=3,strides=2),
        nn.Conv2D(256,kernel_size=5,padding=2,activation='relu'),
        nn.MaxPool2D(pool_size=3,strides=2),
        nn.Conv2D(384,kernel_size=3,padding=1,activation='relu'),
        nn.Conv2D(384,kernel_size=3,padding=1,activation='relu'),
        nn.Conv2D(256,kernel_size=3,padding=1,activation='relu'),
        nn.MaxPool2D(pool_size=3,strides=2),
        #丢弃法
        nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
        nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
        #全连接输出层
        nn.Dense(10))

#随机生成一个矩阵
#并打印每层输出的形状
X=nd.random.uniform(shape=(1,1,224,224))
net.initialize()
for layer in net:
    X=layer(X)
    print(layer.name,'output shape:\t',X.shape)

#加载数据并转换
def load_data_fashion_mnist(batch_size,resize=None,root=os.path.join('~','.mxnet','datasets','fashion-mnist')):
    root=os.path.expanduser(root)
    transformer=[]
    #如果改变图像大小
    if resize:
        transformer+=[gdata.vision.transforms.Resize(resize)]
    transformer+=[gdata.vision.transforms.ToTensor()]
    #将列表里面的元素组合在一起
    transformer=gdata.vision.transforms.Compose(transformer)
    mnist_train=gdata.vision.FashionMNIST(root=root,train=True)
    mnist_test=gdata.vision.FashionMNIST(root=root,train=False)
    num_worker=0 if sys.platform.startswith('win32') else 4
    train_iter=gdata.DataLoader(
        mnist_train.transform_first(transformer),batch_size,shuffle=True,num_workers=num_worker
    )
    test_iter=gdata.DataLoader(
        mnist_test.transform_first(transformer),batch_size,shuffle=False,num_workers=num_worker
    )
    return train_iter,test_iter

batch_size=16
train_iter,test_iter=load_data_fashion_mnist(batch_size=batch_size,resize=224)

lr,num_epochs,ctx=0.05,2,d2l.try_gpu()
net.initialize(force_reinit=True,ctx=ctx,init=init.Xavier())
#训练参数
trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
d2l.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)