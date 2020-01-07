from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import os
from sklearn.decomposition import KernelPCA
from keras.models import Model,Input
from keras.layers import Dense
import tensorflow as tf
import keras.backend as backend



#图片转数组
def img_to_array(img_dir):
    pic_dir=os.listdir(img_dir)
    img_label = []
    img_arr = []
    for filedir in pic_dir:
        img_list = os.listdir(img_dir+'\\'+filedir)

        num_img=0
        for i in img_list:
            num_img+=1
            img_filename=os.path.join((img_dir+'\\'+filedir),i)
            img=image.load_img(img_filename)
            img=img.convert('L')
            img=image.img_to_array(img)
            img=img.squeeze()
            img_arr.append(img)
        if filedir=='经向疵点':
            #print('径向疵点',num_img)
            img_label.append(np.ones(shape=num_img)*1)
            #print(img_label)
        elif filedir=='纬向疵点':
            #print('纬向疵点',num_img)
            img_label.append(np.ones(shape=num_img)*2)
            #print(img_label)
        elif filedir=='块状疵点':
            #print('块状疵点',num_img)
            img_label.append(np.ones(shape=num_img)*3)
            #print(img_label)
        else:
            #print('正常样本',num_img)
            img_label.append(np.ones(shape=num_img)*4)
            #print(img_label)
    #return np.array([img_arr,img_label])
    return img_arr,img_label


#获取数据并转换
train_dir_path=r'D:\code\No4\traindata'
train_feature,train_label=img_to_array(train_dir_path)
#np.random.shuffle(train_data)
train_feature=np.asarray(train_feature)
train_feature=train_feature.reshape(795,256*128)
train_label=np.asarray(train_label)
train_label=np.hstack((train_label[0],train_label[1],train_label[2],train_label[3]))

test_dir_path=r'D:\code\No4\testdata'
test_feature,test_label=img_to_array(test_dir_path)
#np.random.shuffle(train_data)
test_feature=np.asarray(test_feature)
test_label=np.asarray(test_label)
test_feature=test_feature.reshape(len(test_feature),256*128)

test_label=np.hstack((test_label[0],test_label[1],test_label[2],test_label[3]))

#train_data=np.array([train_feature.tolist(),train_label.tolist()]).transpose()
#组合属性与标签
train_data=np.column_stack((train_feature,train_label))
test_data=np.column_stack((test_feature,test_label))
np.random.shuffle(train_data)
np.random.shuffle(test_data)


#决策树分类
dec_model=DecisionTreeClassifier()
# dec_model.fit(train_feature,train_label)
# dec_model.score(test_feature,test_label)
dec_model.fit(train_data[:,:-1],train_data[:,-1])
dec_score=dec_model.score(test_data[:,:-1],test_data[:,-1])
print('决策树分类准确率为:%.4f:'%(dec_score))


#svm分类
svm_model=SVC()
svm_model.fit(train_data[:,:-1],train_data[:,-1])
svm_score=svm_model.score(test_data[:,:-1],test_data[:,-1])
print('SVM分类准确率为:%.4f:'%(svm_score))


#PCA降维
from sklearn.decomposition import PCA
pca=PCA(n_components=0.95)
pca.fit(train_data[:,:-1],train_data[:,-1])
dataset=pca.transform(train_data[:,:-1])
dec_model.fit(dataset,train_data[:,-1])

testdata=pca.transform(test_data[:,:-1])
dec_score1=dec_model.score(testdata,test_data[:,-1])
print('PCA降维之后的决策树分类准确率为:%.4f:'%(dec_score1))

svm_model.fit(dataset,train_data[:,-1])
svm_score1=svm_model.score(testdata,test_data[:,-1])
print('PCA降维之后的SVM分类准确率为:%.4f:'%(svm_score1))

#自动编码器压缩
#自动编码器接收输入不能过大
origimg_datasize=train_data[:,:-1].shape[1]
origimg=train_data[:,:-1]
input_data=Input(shape=(origimg_datasize,))

#编码

origimg=tf.convert_to_tensor(origimg)
origimg=backend.cast(origimg,dtype='float32')


layer1_out=int(origimg_datasize/4)
encode_imgdata=Dense(units=layer1_out,activation='relu')(origimg)
layer2_out=int(layer1_out/4)
encode_imgdata=Dense(units=layer2_out,activation='relu')(encode_imgdata)
layer3_out=int(layer2_out/4)
encode_imgdata=Dense(units=layer3_out,activation='relu')(encode_imgdata)
#解码
decode_imgdata=Dense(units=layer2_out,activation='relu')(encode_imgdata)
decode_imgdata=Dense(units=layer1_out,activation='relu')(decode_imgdata)
decode_imgdata=Dense(units=origimg_datasize,activation='tanh')(decode_imgdata)

autoencoder=Model(inputs=input_data,outputs=decode_imgdata)
encoder=Model(inputs=input_data,outputs=encode_imgdata)

autoencoder.compile(optimizer='adam',loss='mse')

autoencoder.fit(origimg,origimg)
