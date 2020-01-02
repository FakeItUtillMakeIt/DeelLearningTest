#导入相关库
from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd,nd
import random

num_inputs=2
num_example=1000
true_w=[2,-3.4]
true_b=4.2
#训练数据
features=nd.random.normal(scale=1,shape=(num_example,num_inputs))#训练数据集的大小为1000*2
labels=true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b
#加上一定量误差
labels+=nd.random.normal(scale=0.01,shape=labels.shape)
###
def use_svg_display():
    #用矢量图表示
    display.set_matplotlib_formats('jpeg')

def set_figure(figsize=(3.5,2.5)):
    use_svg_display()
    #设置图像大小
    plt.rcParams['figure.figsize']=figsize

set_figure()
plt.scatter(features[:,1].asnumpy(),labels.asnumpy(),1)#加分号只显示图
plt.scatter(features[:,0].asnumpy(),labels.asnumpy(),1,c='red')
plt.show()

#批量读取数据集
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indice=list(range(num_examples))
    random.shuffle(indice)#打乱顺序
    #从中随机取batch_size大小的数据
    for i in range(0,num_examples,batch_size):
        #总是去batch_size大小的数据，如果最后数据不足这个大小则取到末尾
        j=nd.array(indice[i:min(i+batch_size,num_examples)])
        #yield关键字在执行完当前语句之后继续向下执行
        yield features.take(j),labels.take(j)#take函数根据索引返回对应的元素

batch_size=10
for x,y in data_iter(batch_size,features,labels):
    print(x,y)
    break

#给特征参数赋权值，
w=nd.random.normal(scale=0.01,shape=(num_inputs,1))
b=nd.zeros(shape=(1,))

#创建梯度
w.attach_grad()
b.attach_grad()

#定义矢量线性模型
def linreg(X,w,b):
    return nd.dot(X,w)+b

#定义损失函数
def square_loss(y_hat,y):
    return (y_hat-y.reshape(y_hat.shape))**2/2

#定义优化算法
def sgd(params,lr,batch_size):
    for param in params:
        param[:]=param-lr*param.grad/batch_size

#训练模型
lr=0.03
num_epochs=3
loss=square_loss
for epoch in range(num_epochs):
    for x,y in data_iter(batch_size,features,labels):
        with autograd.record():
            lo=loss(linreg(x,w,b),y)
        lo.backward()
        sgd([w,b],lr,batch_size)
    train_l=loss(linreg(features,w,b),labels)
    #输出每一次更新的损失
    print('epoch%d,loss%f'%(epoch+1,train_l.mean().asnumpy()))


