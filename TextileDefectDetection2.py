import time
import os
import PIL.Image as Image
import numpy as np
import tensorflow
import tensorflow.keras as keras


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
    print(path)
    train_num=0
    test_num=0

    if path=='traindata':
        print(dataset_path)
        #yield
        trainlabel=[]
        traindata=[]
    elif path=='testdata':
        print(dataset_path)
        #yield
        testlabel=[]
        testdata=[]
    else:
        break
    imgfile=os.listdir(dataset_path)

    for class_img in imgfile:
        print(class_img)
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

trainx=trainset[:,:-1].reshape(795,128,256)
testx=testset[:,:-1].reshape(457,128,256)

#卷积神经  11 weight layers
#层返回
def conv_poollayer(layers=1,conv_channels=1,conv_size=3,pool_size=2,pool_type='max'):
    net=tensorflow.keras.Sequential()
    convnet = tensorflow.keras.layers.Conv2D(filters=conv_channels,kernel_size=conv_size,
                                             strides=1)
    for i in range(layers):
        net.add(convnet)
    if pool_type=='max':
        poolnet=tensorflow.keras.layers.MaxPool2D(pool_size=pool_size)
    else:
        poolnet=tensorflow.keras.layers.AveragePooling2D(pool_size=pool_size)
    net.add(poolnet)
    return net



convnet=tensorflow.keras.Sequential()
convnet.add(tensorflow.keras.layers.Conv2D())
