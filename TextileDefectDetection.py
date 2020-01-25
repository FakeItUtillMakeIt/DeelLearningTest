#纺织品瑕疵检测
import time
import os
import PIL.Image as Image
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

#导入文件并分类
imgfile_path=os.path.abspath('..\\No4')
#imgfile_path=r'D:\code\No4'#包括训练集和测试集在内的所有图片
classfile=os.listdir(imgfile_path)
traindata=[]
testdata=[]
trainlabel=[]
testlabel=[]
for path in classfile:
    dataset_path=imgfile_path+'\\'+path
    #print(path)
    train_num=0
    test_num=0

    if path=='traindata':
        #print(dataset_path)
        #yield
        trainlabel=[]
        traindata=[]
    elif path=='testdata':
        #print(dataset_path)
        #yield
        testlabel=[]
        testdata=[]
    else:
        break
    imgfile=os.listdir(dataset_path)

    for class_img in imgfile:
        #print(class_img)
        class_imgpath=dataset_path+'\\'+class_img
        if class_img == '经向疵点':
            classify=1
        elif class_img=='纬向疵点':
            classify=2
        elif class_img == '块状疵点':
            classify = 3
        elif class_img == '正常样本':
            classify = 4

        #进入img文件
        img_filelist=os.listdir(class_imgpath)
        for img_file in img_filelist:
            img=class_imgpath+'\\'+img_file

            if path == 'traindata':
                imgarr=Image.open(img)
                imgarr=np.array(imgarr)
                imgarr=imgarr.flatten()
                imgarr=imgarr.tolist()
                traindata.append(imgarr)
                #traindata=np.array(traindata)
                trainlabel.append(classify)
            elif path=='testdata':
                imgarr = Image.open(img)
                imgarr = np.array(imgarr)
                imgarr = imgarr.flatten()
                imgarr=imgarr.tolist()
                testdata.append(imgarr)
                #testdata = np.array(testdata)
                testlabel.append(classify)
            else:
                break
        #yield

traindata=np.array(traindata,dtype='int8')
testdata=np.array(testdata,dtype='int8')
trainlabel=np.array(trainlabel,dtype='int8')
testlabel=np.array(testlabel,dtype='int8')
#得到训练集和测试集
trainset=np.column_stack((traindata,trainlabel))
testset=np.column_stack((testdata,testlabel))
#将其样本随机化
np.random.shuffle(trainset)
np.random.shuffle(testset)


#调参
#原数据训练SVM测试
from sklearn.svm import NuSVC
start_time=time.time()
svm_model=NuSVC(kernel='poly',tol=1e-3)#核函数
#svm_model.fit(trainset[:,:-1],trainset[:,-1])
svm_model.fit(trainset[:,:-1],trainset[:,-1])
svm_score=svm_model.score(testset[:,:-1],testset[:,-1])
end_time=time.time()
print('原数据集SVM测试准确率:%.3f,时间花费为:%.3f'%(svm_score,(end_time-start_time)))

#原数据决策树测试
start_time=time.time()
dec_model=DecisionTreeClassifier()
dec_model.fit(trainset[:,:-1],trainset[:,-1])
dec_score=dec_model.score(testset[:,:-1],testset[:,-1])
end_time=time.time()
print('原数据集决策树测试准确率:%.3f,时间花费为:%.3f'%(dec_score,(end_time-start_time)))

#原数据集成学习
start_time=time.time()
adaboost_model=AdaBoostClassifier(base_estimator=svm_model,learning_rate=0.05,algorithm='SAMME',)
adaboost_model.fit(trainset[:,:-1],trainset[:,-1])
adaboost_score=adaboost_model.score(testset[:,:-1],testset[:,-1])
end_time=time.time()
print('原数据集集成SVM测试准确率:%.3f,时间花费为:%.3f'%(adaboost_score,(end_time-start_time)))


#集成学习，多模型
from sklearn.ensemble import RandomForestClassifier
start_time=time.time()
rdforest=RandomForestClassifier()
vote_model=VotingClassifier(estimators=[('rf',rdforest),('sm',svm_model),('dm',dec_model)])
vote_model.fit(trainset[:,:-1],trainset[:,-1])
vote_score=vote_model.score(testset[:,:-1],testset[:,-1])
end_time=time.time()
print('原数据集多集成模型测试准确率:%.3f,时间花费为:%.3f'%(vote_score,(end_time-start_time)))

#PCA降维
pca=PCA(n_components=0.95)
#pca=PCA()
pca.fit(trainset[:,:-1])
trainX=pca.transform(trainset[:,:-1])
testX=pca.transform(testset[:,:-1])


#数据规范化
start_time=time.time()
nor=Normalizer()
nor.fit(trainX)
trainX=nor.transform(trainX)
testX=nor.transform(testX)
dec_model.fit(trainX,trainset[:,-1])
dec_score=dec_model.score(testX,testset[:,-1])
svm_model.fit(trainX,trainset[:,-1])
svm_score=svm_model.score(testX,testset[:,-1])
adaboost_model.fit(trainX,trainset[:,-1])
adaboost_score=adaboost_model.score(testX,testset[:,-1])
vote_model.fit(trainX,trainset[:,-1])
vote_score=vote_model.score(testX,testset[:,-1])
end_time=time.time()
print('处理数据集SVM测试准确率:%.3f,时间花费为:%.3f'%(svm_score,(end_time-start_time)))
print('处理数据集决策树测试准确率:%.3f,时间花费为:%.3f'%(dec_score,(end_time-start_time)))
print('处理数据集adaboost测试准确率:%.3f,时间花费为:%.3f'%(adaboost_score,(end_time-start_time)))
print('处理数据集vote测试准确率:%.3f,时间花费为:%.3f'%(vote_score,(end_time-start_time)))

#全连接神经网络
start_time=time.time()
nn_model=MLPClassifier(solver='adam',hidden_layer_sizes=1000,learning_rate_init=0.005)
nn_model.shuffle=True
nn_model.batch_size=128
nn_model.activation='tanh'
nn_model.fit(trainX,trainset[:,-1])
nn_score=nn_model.score(testX,testset[:,-1])
end_time=time.time()
print('处理数据集神经网络测试准确率:%.3f,时间花费为:%.3f'%(nn_score,(end_time-start_time)))





