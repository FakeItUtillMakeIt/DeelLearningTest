from keras.preprocessing import image
import numpy as np
from pywt import dwt2
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pywt

file=r'./51.jpg'
img=cv2.imread(file)
#多通道变为单通道
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)

plt.figure("二维小波一级变换")
coeffs=pywt.dwt2(img,'haar')
ca,(cb,cc,cd)=coeffs

plt.subplot(221)
plt.imshow(ca,'gray')
plt.subplot(222)
plt.imshow(cb,'gray')
plt.subplot(223)
plt.imshow(cc,'gray')
plt.subplot(224)
plt.imshow(cd,'gray')

plt.show()

