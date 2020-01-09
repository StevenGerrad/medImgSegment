
###################################################################################
#
#       2019.7.15
#       医学脑血管图像分割
#       脑内容分割
#       -----------
#       前面的都不太靠谱啊，试一下不通过头骨分割直接仍借鉴分水岭，采用直接腐蚀脑内物质
# 
###################################################################################


import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import string
import pylab


def pltImgShow(image):
    plt.title('pltImgShow')
    plt.imshow(image)
    plt.show()

img1 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0061.tif"
img2 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0073.tif"
img3 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0085.tif"
img4 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0097.tif"
img5 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0109.tif"
img6 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0121.tif"
img7 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0133.tif"
img8 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0145.tif"
img9 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0157.tif"
img10 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0169.tif"

img = cv.imread('D:/auxiliaryPlane/project/Python/medImgSegment/imgData/'+img5)
plt.subplot(2,3,1)
plt.title('original')
plt.imshow(img)

# 灰度图，但其实原来的图片的RGB通道的值都相同
img_gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
plt.subplot(2,3,2)
plt.title('img_gray')
plt.imshow(img_gray)
# 二值图像
ret, thresh = cv.threshold(img_gray,254,255,cv.THRESH_BINARY)
# thresh = cv.bitwise_not(thresh, thresh)
plt.subplot(2,3,3) 
plt.title('thresh')
plt.imshow(thresh)
# 形态学变换，opening用3x3先腐蚀再膨胀，去掉个别点
kernel = np.ones((5,5),np.uint8) 
# 观察发现基本上3x3需要5次操作，5x5需要3次操作
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 3)
# opening = cv.morphologyEx(opening,cv.MORPH_CLOSE,kernel, iterations = 3)

# kernel1 = np.ones((7,7),np.uint8) 
# opening = cv.erode(opening,kernel1,iterations=7)

dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)   
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,cv.THRESH_BINARY)
sure_fg = np.uint8(sure_fg)
# unknown = cv.subtract(sure_bg,sure_fg)

# 我佛了，这个腐蚀倒是很清楚不要破坏连通性
print(opening.shape)
plt.subplot(2,3,4)
plt.title('opening')
plt.imshow(opening)

sure_bg = cv.dilate(opening,kernel,iterations=3)
sure_bg = cv.bitwise_not(sure_bg,sure_bg)       # 将二值图像的值反转
plt.subplot(2,3,5)
plt.title('sure_bg')
plt.imshow(sure_bg)
# 形态变换后基本上三次膨胀闭口就贴上了

nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(sure_bg)
markers = labels
print(markers.shape)
markers = markers+1
img_gray[markers != 3] = 255
plt.subplot(2,3,6), plt.title('markers'), plt.imshow(markers)

plt.show()

markers = cv.watershed(img, markers)
img[markers == -1] = [255,0,0]
plt.subplot(1,2,1); plt.title('img'); plt.imshow(img)
plt.subplot(1,2,2); plt.title('markers'); plt.imshow(markers)
plt.show()






