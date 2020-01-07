import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model  # 泛型模型
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


# X shape (60,000 28x28), y shape (10,000, )
(x_train, _), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255. - 0.5  # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5  # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_train.shape)
print(x_test.shape)

# 压缩特征维度至2维
encoding_dim = 2

# this is our input placeholder
input_img = Input(shape=(784,))

# 编码层
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(10, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)
#encoder_output=Dense(encoding_dim,activation='relu')(input_img)


# 解码层
decoded = Dense(10, activation='relu')(encoder_output)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)

#decoded=Dense(784,activation='tanh')(encoder_output)
# 构建自编码模型
autoencoder = Model(inputs=input_img,outputs=decoded)

# 构建编码模型
encoder = Model(inputs=input_img, outputs=encoder_output)
#构建解码模型
#decoder = Model(inputs=encoder_output,outputs=decoded)
# compile autoencoder
#optimizers参数指定梯度更新方法'sgd,adam等
autoencoder.compile(optimizer='adam', loss='mse')

# training
autoencoder.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True)

# plotting
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, s=3)
plt.colorbar()
plt.show()

#压缩后
clf=DecisionTreeClassifier(max_depth=10,random_state=10)
clf.fit(encoded_imgs[:5000],y_test[:5000])
score1=clf.score(encoded_imgs[5000:],y_test[5000:])

#无压缩
clf2=DecisionTreeClassifier(max_depth=10,random_state=10)
clf2.fit(x_test[:5000],y_test[:5000])
score2=clf2.score(x_test[5000:],y_test[5000:])

print('score1 is :%f,score2 is :%f.'%(score1,score2))

#自动编码器降维对数据量不能太大