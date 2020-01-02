#逻辑回归

import numpy
import theano
import theano.tensor as T
import sklearn.metrics
import time

def l2(x):
    return T.sum(x**2)

examples=1000
features=100
training_steps=1000

#生成数据集,每组数据为100维，做二分类
D=(numpy.random.randn(examples,features),numpy.random.randint(size=examples,low=0,high=2))

x=T.dmatrix('x')
y=T.dvector('y')
w=theano.shared(numpy.random.randn(features),name='w')#随机生成初始w
b=theano.shared(0.,name='b')

p=1/(1+T.exp(-T.dot(x,w)-b))#

error=T.nnet.binary_crossentropy(p,y)#采用二次交叉熵作为损失函数
loss=error.mean()+0.01*l2(w)#损失的均值+二次正则项

prediction=p>0.5

#loss=T.sum(loss)
#loss必须是标量
gw,gb = T.grad(loss,[w,b])

train=theano.function(inputs=[x,y],outputs=[p,error],updates=((w,w-0.1*gw),(b,b-0.1*gb)))
predict=theano.function(inputs=[x],outputs=prediction)

print(sklearn.metrics.accuracy_score(D[1],predict(D[0])))

start_time=time.time()
for i in range(training_steps):
    prediction,error=train(D[0],D[1])
end_time=time.time()

print(sklearn.metrics.accuracy_score(D[1],predict(D[0])))
print('花费时间为:',end_time-start_time)


