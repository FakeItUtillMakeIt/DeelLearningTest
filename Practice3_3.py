#protect3_2的更简洁实现,使用框架实现

from mxnet import autograd,nd
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon

#生成数据集
num_examples=1000
num_inputs=2
true_w=[2,-3.4]
true_b=4.2
features=nd.random.normal(scale=1,shape=(num_examples,num_inputs))
labels=true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
labels+=labels+nd.random.normal(scale=0.01,shape=labels.shape)

#读取数据集
#批量大小为10
batch_size=10
#将特征和标签组合在一起
dataset=gdata.ArrayDataset(features,labels)
#生成随机小批量迭代器
data_iter=gdata.DataLoader(dataset,batch_size,shuffle=True)

#定义模型
#定义一个Seuential容器，该容器串联神经网络各个层
net=nn.Sequential()
#神经网络中Dense实例是一个全连接层，
#下面定义一个全连接层，输出层为一个输出
net.add(nn.Dense(1))

#初始化模型参数
#init库定义了模型参数初始化的各种方法
net.initialize(init.Normal(sigma=0.01))

#定义损失函数
#定义平凡损失函数
loss=gloss.L2Loss()

#定义优化算法
trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})

#训练模型
num_epochs=3
for epoch in range(num_epochs):
    for x,y in data_iter:
        with autograd.record():
            lo=loss(net(x),y).mean()#lo=loss(net(x),y)
        lo.backward()
        trainer.step(1) #trainer.step(batch_size)
    lo=loss(net(features),labels)
    print('epoch%d,loss%f'%(epoch,lo.mean().asnumpy()))


