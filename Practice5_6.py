#AlexNet卷积神经网络

from sklearn.decomposition import PCA
from keras.preprocessing import image
import os
import numpy as np
import matplotlib.pyplot as plt


imgdir=r'D:\code\No4\traindata\经向疵点\51.bmp'
img1=image.load_img(imgdir)
img=img1.convert('L')
img_data=image.img_to_array(img)
img_data=np.squeeze(img_data)
pca=PCA(n_components=0.9)
pca.fit(img_data)
pca_imgdata=pca.transform(img_data)
plt.subplot(1,2,1)
a=plt.imshow(img_data)
plt.subplot(1,2,2)
b=plt.imshow(pca_imgdata)
plt.show()

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


