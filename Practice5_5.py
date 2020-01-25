#经典卷积神经网络LeNET

import d2lzh as d2l
import mxnet as mx
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss,nn
import time
import tensorflow as tf

#LeNet模型网络
net=nn.Sequential()
net.add(nn.Conv2D(channels=6,kernel_size=5,activation='sigmoid'),
        nn.MaxPool2D(pool_size=2),
        nn.Conv2D(channels=6,kernel_size=5,activation='sigmoid'),
        nn.MaxPool2D(pool_size=2),
        nn.Dense(units=120,activation='sigmoid'),
        nn.Dense(units=84,activation='sigmoid'),
        nn.Dense(units=10,activation='sigmoid')
        )

# (trainx,trainy),(testx,testy)=tf.keras.datasets.mnist.load_data()
# trainx,testx=trainx.reshape(60000,784),testx.reshape(10000,784)
# train_data1=trainx[1].reshape((1,1,28,28))

data=train_data=nd.random.uniform(shape=(1,1,28,28))
net.initialize()
for layer in net:
        train_data=layer(train_data)
        print(layer.name,'out_shape',train_data.shape)

#训练数据
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size)

def try_gpu():
        try:
                ctx=mx.gpu()
                _=nd.zeros((1,),ctx=ctx)
        except mx.base.MXNetError:
                ctx=mx.cpu()
        return ctx

ctx=try_gpu()
#ctx=mx.gpu()

#计算模型准确率
def evaluate_accuracy(data_iter,net,ctx):
        acc_sum,n=nd.array([0],ctx=ctx),0
        for X,y in data_iter:
                #如果ctx代表GPU及相应的显存，则将数据复制到显存
                X,y=X.as_in_context(ctx),y.as_in_context(ctx).astype('float32')
                acc_sum+=(net(X).argmax(axis=1)==y).sum()
                n+=y.size
        return acc_sum.asscalar()/n

#计算损失函数，更新网络参数
def train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs):
        print('training on',ctx)
        loss=gloss.SoftmaxCrossEntropyLoss()
        for epoch in range(num_epochs):
                train_l_sum,train_acc_sum,n,start=0.0,0.0,0,time.time()
                for X,y in train_iter:
                    #as_in_context函数使目标变量和原变量共享内存或显存
                        X,y=X.as_in_context(ctx),y.as_in_context(ctx)
                        with autograd.record():
                                y_hat=net(X)
                                l=loss(y_hat,y).sum()
                        l.backward()
                        #更新batch_size批量的参数
                        trainer.step(batch_size)
                        y=y.astype('float32')
                        train_l_sum+=l.asscalar()
                        train_acc_sum+=(y_hat.argmax(axis=1)==y).sum().asscalar()
                        n+=y.size
                test_acc=evaluate_accuracy(test_iter,net,ctx)
                print('epoch%d,loss%.4f,train acc%.3f,test_acc%.3f,time%.1f sec'%(epoch+1,train_l_sum/n,
                                                                                  train_acc_sum/n,test_acc,time.time()-start))
lr,num_epochs=0.9,50
net.initialize(force_reinit=True,ctx=ctx,init=init.Xavier())
#更新参数函数
trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)

