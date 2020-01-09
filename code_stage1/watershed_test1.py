
###################################################################################
#
#       2019.7.2
#       医学脑血管图像分割
#       分水岭算法
#       https://segmentfault.com/a/1190000015690356
# 
###################################################################################


# Otsu的二值化找到的近似估计值.

import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = 'DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0037.tif'

img = cv2.imread('D:/auxiliaryPlane/project/Python/medImgSegment/imgData/'+img1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.namedWindow("show",cv2.WINDOW_NORMAL)
cv2.resizeWindow("show", 512, 512)
cv2.imshow('show',thresh)

# noise removal
# 现在我们需要去除图像中的任何小的白噪声,因此我们要使用形态学开运算
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
# 对输入图像进行膨胀操作
sure_bg = cv2.dilate(opening,kernel,iterations=3)   

# Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv.DIST_L2,5)
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

# 标记已经准备好了,现在是最后一步的时候了，应用分水岭.
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]


cv2.namedWindow("enhanced",cv2.WINDOW_NORMAL)
cv2.resizeWindow("enhanced", 512, 512)
cv2.imshow('enhanced',img)
cv2.waitKey(0)
cv2.destroyAllWindows()