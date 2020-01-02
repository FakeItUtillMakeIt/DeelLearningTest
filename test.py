from IPython import display
from mxnet import nd,autograd
from matplotlib import pyplot as plt
import random


num_inputs=2
num_example=1000
true_w=[2,-3.4]
true_b=4.2
bacth_size=10

features=nd.random.normal(scale=1, shape=(num_example,num_inputs))
#真实标签
labels=true_w[0]*features[:, 0]+true_w[1]*features[:, 1]+true_b
#预测标签
labels+=nd.random.normal(scale=0.01, shape=labels.shape)

def use_svg_display():
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(5,4)):
    use_svg_display()
    #设置图的尺寸
    plt.rcParams['figure.figsize']=figsize

set_figsize()
plt.scatter(features[:,0].asnumpy(),labels.asnumpy(),1)
plt.scatter(features[:,1].asnumpy(),labels.asnumpy(),1)

#plt.show()

#定义一个返回批量随机特征和标签的样本
def data_iter(bacth_size,features,labels):
    num_example=len(features)
    induices=list(range(num_example))
    random.shuffle(induices)#随机化样本读取顺序
    for i in range(0,num_example,bacth_size):
        j=nd.array(induices[i:min(i+bacth_size,num_example)])
        yield features.take(j),labels.take(j)#take函数根据索引返回元素

#输出一组特征和标签
for X,y in data_iter(10,features,labels):
    print(X,y)
    break

w=nd.random.normal(scale=0.01,shape=(num_inputs,1))#权重
b=nd.zeros(shape=(1,1))#偏差

#申请存储梯度所需要的内存
w.attach_grad()
b.attach_grad()

#定义模型
def linreg(X,w,b):
    return nd.dot(X,w)+b

#定义损失函数
def squared_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2#将真实值的形状变为预测值的形状

def sgd(params,lr,batch_size):#小批量梯度算法，不断迭代模型参数来优化损失函数，
    #这里的params为需要求得的特征参数权值与常数项
    for param in params:# 自动求梯度得到的是样本批量梯度和，除以样本数得到平均梯度
        param[:]=param-lr*param.grad/batch_size

lr=0.03
num_epochs=3#迭代次数
net=linreg#使用线性回归
loss=squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(bacth_size,features,labels):
        with autograd.record():
            l=loss(net(X,w,b),y)
        l.backward()
        sgd([w, b], lr, bacth_size)
        train_l=loss(net(features,w,b),labels)
        print('epoch %d,loss %f'% (epoch+1,train_l.mean().asnumpy()))

