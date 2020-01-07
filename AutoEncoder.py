#自动编码器

import numpy as np

np.random.seed(100)

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense,Input
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier


(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.astype('float32')/255-0.5
x_test=x_test.astype('float32')/255-0.5
x_train=x_train.reshape((x_train.shape[0],-1))
x_test=x_test.reshape((x_test.shape[0],-1))

print(x_train.shape)
print(x_test.shape)

#压缩成2维
encoding_dim=2

input_img=Input(shape=(784,))

#编码层
encode_output=Dense(units=encoding_dim,activation='relu')(input_img)

#解码层
decode_output=Dense(units=x_test.shape[1],activation='tanh')(encode_output)

#构建自编码模型
autoencoder=Model(inputs=input_img,outputs=decode_output)

#构建编码模型
encoder=Model(inputs=input_img,outputs=encode_output)
#构建解码模型
#decoder=Model(inputs=encode_output,outputs=decode_output)

#编译自编码模型
autoencoder.compile(optimizer='adam',loss='mse')
#训练
autoencoder.fit(x_train,x_train,epochs=20,batch_size=256,shuffle=True)

#测试
encode_test=encoder.predict(x_test)

plt.scatter(encode_test[:,0],encode_test[:,1],c=y_test,s=3)
plt.colorbar()
plt.show()

#压缩后
clf=DecisionTreeClassifier(max_depth=10,random_state=10)
clf.fit(encode_test[:5000],y_test[:5000])
score1=clf.score(encode_test[5000:],y_test[5000:])

#无压缩
clf2=DecisionTreeClassifier(max_depth=10,random_state=10)
clf2.fit(x_test[:5000],y_test[:5000])
score2=clf2.score(x_test[5000:],y_test[5000:])

print('score1 is :%f,score2 is :%f.'%(score1,score2))


