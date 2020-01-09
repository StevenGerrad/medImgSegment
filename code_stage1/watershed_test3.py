
###################################################################################
#
#       2019.7.8
#       医学脑血管图像分割
#       分水岭
#       https://blog.csdn.net/TingHW/article/details/84578541
# 
###################################################################################

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import string
import pylab

img1 = "imgData/DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0061.tif"
img2 = "coins.jpg"

# img = cv.imread('D:/auxiliaryPlane/project/Python/medImgSegment/imgData/'+img1)
# plt.title('oriange')
# plt.imshow(img,'gray')
# plt.show()

# 常用的硬币示例图
img = cv.imread('D:/auxiliaryPlane/project/Python/medImgSegment/'+img2)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

print(thresh.shape)
plt.subplot(2,3,1)
plt.title('gray')
plt.imshow(thresh,'gray')

# noise removal
kernel = np.ones((3,3),np.uint8)        # 3x3矩阵，所有元素为1
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)      # 形态学变换

plt.subplot(2,3,2)
plt.title('morphologyEx')
plt.imshow(opening)

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)     # 计算图像中每一个非零点距离离自己最近的零点的距离
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
print(sure_fg.dtype)

# Finding unknown region
sure_fg = np.uint8(sure_fg)     # 数据类型转化
print(sure_fg.dtype)
unknown = cv.subtract(sure_bg,sure_fg)      # 计算数组间的元素差

plt.subplot(2,3,3)
plt.title('unknown')
plt.imshow(unknown)

# 现在我们可以确定哪些是硬币的区域，哪些是背景，哪些是背景.因此，我们创建标记
#  (它是一个与原始图像相同大小的数组，但使用int32数据类型)并对其内部的区域进行标记.
# 我们知道，如果背景是0，那么分水岭将会被认为是未知的区域, 所以我们用不同的整数来标记它,用0表示由未知定义的未知区域.

# Marker labelling 将图像的背景标记为0，然后其他对象从1开始标记为整数.
ret, markers = cv.connectedComponents(sure_fg)
print(markers.shape)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255]=0
plt.subplot(2,3,4)
plt.title('final sure_fg')
plt.imshow(sure_fg)

markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]

plt.subplot(2,3,5)
plt.title('img')
plt.imshow(img)

plt.subplot(2,3,6)
plt.title('markers')
plt.imshow(markers)

plt.show()
