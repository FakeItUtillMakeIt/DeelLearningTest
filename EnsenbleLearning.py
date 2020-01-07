
#集成学习

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from sklearn.svm import SVC

(trainx,trainy),(testx,testy)=tf.keras.datasets.mnist.load_data()

trainx=trainx.reshape((60000,784))
testx=testx.reshape((10000,784))
#单个模型集成学习
ada_boost=AdaBoostClassifier(DecisionTreeClassifier())
ada_boost.fit(trainx[:1000],trainy[:1000])

print(ada_boost.score(testx,testy))

#SVM集成学习
#使用SVM时需要设置probability
svm_boost=AdaBoostClassifier(SVC(probability=True,kernel='linear')e)
svm_boost.fit(trainx[:100],trainy[:100])

#多个模型集成学习
from sklearn.ensemble import VotingClassifier


vote_cls=VotingClassifier(estimators=[
    ("svm_cla",SVC()),
    ("tree",DecisionTreeClassifier())
],voting='hard'
)

vote_cls.fit(trainx[:1000],trainy[:1000])
vote_cls.score(testx,testy)

#留出法
from sklearn.model_selection import train_test_split

train_X,train_Y,test_X,test_Y=train_test_split(trainx,trainy,test_size=0.3,random_state=666)

from sklearn.model_selection import cross_validate
#交叉验证
cv_result=cross_validate(estimator=SVC(),X=train_X[:1000],y=test_X[:1000],cv=10)