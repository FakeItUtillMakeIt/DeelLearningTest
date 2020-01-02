#手写字符识别

import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.svm as SVM
import sklearn.tree as st
import sklearn.metrics as sm
import graphviz

(train_x,train_y),(test_x,test_y)=tf.keras.datasets.mnist.load_data()
#clf=SVM.SVC()
train_x=train_x.reshape((60000,784))
test_x=test_x.reshape((10000,784))

accsum=[]

for i in range(100):
    clf=st.DecisionTreeClassifier(max_depth=i+1)
    clf.fit(train_x[:1000],train_y[:1000])
    y_pred=clf.predict(test_x[:100])
    acc=sm.accuracy_score(test_y[:100],y_pred).ravel()
    print(acc)
    accsum.append((list(acc)).pop())

plt.plot(list(range(100)),accsum)

num_1=train_x[78].reshape((28,28))
#plt.imshow(num_1)
plt.show()


