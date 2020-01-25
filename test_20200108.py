#使用mnist数据集做卷积
from tensorflow.keras.datasets.mnist import load_data
#from keras.datasets.mnist import load_data
import d2lzh as d2l
from mxnet.gluon import nn
from mxnet import nd,gluon,init




(train_fiture,train_label),(test_fiture,test_label)=load_data()


convnet=nn.Sequential()
convnet.add(
            nn.Conv2D(channels=2,kernel_size=3,padding=1,activation='relu'),
            nn.Conv2D(channels=2,kernel_size=3,activation='relu'),
            nn.MaxPool2D(pool_size=2),
            nn.Conv2D(channels=1,kernel_size=2,activation='relu'),
            nn.Dense(units=10,activation='tanh')
            )

lr,num_epochs,batch_size,ctx=0.05,5,128,d2l.try_gpu()
convnet.initialize(init=init.Xavier(),ctx=ctx)
trainer=gluon.Trainer(convnet.collect_params(),'sgd',{'learning_rate':lr})
train_iter,test_iter=(train_fiture,train_label),(test_fiture,test_label)
d2l.train_ch5(convnet,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)