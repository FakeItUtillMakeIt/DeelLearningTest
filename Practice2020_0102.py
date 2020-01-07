from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
import sklearn.neural_network as nn
import tensorflow as tf

(trainx,trainy),(testx,testy)=tf.keras.datasets.mnist.load_data()
trainx,testx=trainx[:1000].reshape(1000,784),testx[:1000].reshape(1000,784)



clf=MLPClassifier(verbose=0,activation='relu',alpha=0.001,batch_size=128,beta_1=0.3)
grid=GridSearchCV(verbose=1,estimator=clf,param_grid={'C':[0.1,0.2,0.3],'learning_rate_init':[0.01,0.02,0.03]})

clf.fit(trainx,trainy[:1000])

