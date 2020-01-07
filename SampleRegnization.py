#from scipy.misc import imread
from PIL import Image
import numpy as np
from keras import Sequential
from keras import Model,Input
from keras.layers import Dense
import pickle
import  tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import sklearn.neural_network as nn


import os

#加载图片并转换
def loadImage(img_loc,h,w):
    im=Image.open(img_loc,'r')
    #im.show()
    im=im.convert('L')
    data=im.getdata()
    data=np.matrix(data)
    #对图像进行缩放
    data=np.resize(data,(h,w))
    new_im=Image.fromarray(data)
    #new_im.show()
    return new_im


# def load_data(file_dir):
#     data,label=pickle.load(open(file_dir,'rb'))
#     return data,label

h,w=32,64
a=loadImage(r'D:\code\No4\traindata\经向疵点\51.bmp',h,w)

#将图片转为矩阵
def batch_imgtomat(root_dir,h,w):
    #root_dir=r'D:\code\No4\traindata\纬向疵点'
    list=os.listdir(root_dir)
    array=[]
    n=0
    for i in range(len(list)):
        path=os.path.join(root_dir,list[i])
        label_c=root_dir[-4:]
        if os.path.isfile(path):
            #print(path)
            n+=1
            img_new=loadImage(path,h,w)
            array.append(np.array(np.matrix(img_new).reshape(h*w,)))
            #array.append((np.matrix(img_new)).tolist())
    if label_c=='经向疵点':
        label=np.ones(shape=n)
    elif label_c=='纬向疵点':
        label=np.ones(shape=n)*2
    elif label_c=='块状疵点':
        label = np.ones(shape=n) * 3
    else:
        label = np.ones(shape=n) * 4
    return (np.array(array)).reshape(n,w*h),label

train_root_dir1=r'D:\code\No4\traindata\经向疵点'
train_root_dir2=r'D:\code\No4\traindata\纬向疵点'
train_root_dir3=r'D:\code\No4\traindata\块状疵点'
train_root_dir4=r'D:\code\No4\traindata\正常样本'

test_root_dir1=r'D:\code\No4\testdata\经向疵点'
test_root_dir2=r'D:\code\No4\testdata\纬向疵点'
test_root_dir3=r'D:\code\No4\testdata\块状疵点'
test_root_dir4=r'D:\code\No4\testdata\正常样本'

#xx_train,x_test类型都是矩阵类型
x_train_1,y_train_1=batch_imgtomat(train_root_dir1,h,w)
x_train_2,y_train_2=batch_imgtomat(train_root_dir2,h,w)
x_train_3,y_train_3=batch_imgtomat(train_root_dir3,h,w)
x_train_4,y_train_4=batch_imgtomat(train_root_dir4,h,w)

x_test_1,y_test_1=batch_imgtomat(test_root_dir1,h,w)
x_test_2,y_test_2=batch_imgtomat(test_root_dir2,h,w)
x_test_3,y_test_3=batch_imgtomat(test_root_dir3,h,w)
x_test_4,y_test_4=batch_imgtomat(test_root_dir4,h,w)

#整合数据
train_feature=np.vstack((x_train_1,x_train_2,x_train_3,x_train_4))
train_label=np.hstack((y_train_1,y_train_2,y_train_3,y_train_4))

test_feature=np.vstack((x_test_1,x_test_2,x_test_3,x_test_4))
test_label=np.hstack((y_test_1,y_test_2,y_test_3,y_test_4))

#特征+标签+随机
train_data=np.array([train_feature.tolist(),train_label.tolist()])
train_data=train_data.transpose()
np.random.shuffle(train_data)


#使用决策树进行分类
dec_model=DecisionTreeClassifier()
dec_model.fit(train_data[:,0].tolist(),train_data[:,1].tolist())
dec_score=dec_model.score(test_feature,test_label)
print('决策树准确率为:%.3f'%(dec_score))


#使用SVM进行分类
svm_model=SVC()
svm_model.fit(train_data[:,0].tolist(),train_data[:,1].tolist())
svm_score=svm_model.score(test_feature,test_label)
print('SVM准确率为:%.3f'%(svm_score))

#使用神经网络进行分类
#先进行自编码器压缩
nn_model=nn.MLPClassifier(activation='relu')
nn_model.fit(train_data[:,0].tolist(),train_data[:,1].tolist())
nn_score=nn_model.score(test_feature,test_label)
print('神经网络准确率为:%.3f'%(nn_score))


input_img=Input(shape=(h*w,))
#编码器
encoder=Dense(units=int(np.ceil(h*w/2)),activation='relu')(input_img)
encoder=Dense(units=int(np.ceil(h*w/8)),activation='relu')(encoder)
encoder=Dense(units=int(np.ceil(h*w/16)),activation='relu')(encoder)
encoder_outputs=Dense(units=int(np.ceil(h*w/32)),activation='relu')(encoder)

#解码器
decoder=Dense(units=int(np.ceil(h*w/16)),activation='relu')(encoder_outputs)
decoder=Dense(units=int(np.ceil(h*w/8)),activation='relu')(decoder)
decoder=Dense(units=int(np.ceil(h*w/2)),activation='relu')(decoder)
decoder_outputs=Dense(units=h*w,activation='tanh')(decoder)

#自动编码器
autoencoder=Model(inputs=input_img,outputs=decoder_outputs)

#编码
encode_ouputs=Model(inputs=input_img,outputs=encoder_outputs)

autoencoder.compile(optimizer='sgd',loss='mse')

#np.random.shuffle(train_feature)
autoencoder.fit(train_feature,train_feature,batch_size=256,epochs=20)
test_encode=encode_ouputs.predict(test_feature)
svm_model.fit(test_encode,train_label)

test_score=svm_model.score(train_feature,train_label)
print(test_score)

