
###################################################################################
#
#       2019.7.13
#       医学脑血管图像分割
#       脑内容分割
#           1. 分水岭形态学操作--膨胀头骨形成闭环，分离
#           2. (1)+均值滤波处理，分离
#           3. 轮廓检测，凸包函数
#           4. 哈夫圆变换，寻找图中的圆形
#           5. 交互式图像分割（利用*1中的部分结果）
#           6. 投影和雷登变换
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
img9 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0169.tif"

img = cv.imread('D:/auxiliaryPlane/project/Python/medImgSegment/imgData/'+img1)
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
plt.subplot(2,3,3) 
plt.title('thresh')
plt.imshow(thresh)
# 形态学变换，opening用3x3先腐蚀再膨胀，去掉个别点
kernel = np.ones((5,5),np.uint8) 
# 观察发现基本上3x3需要5次操作，5x5需要3次操作
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 3)
opening = cv.morphologyEx(opening,cv.MORPH_CLOSE,kernel, iterations = 3)
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

plt.subplot(2,3,6)
plt.title('img_gray')
plt.imshow(img_gray)

plt.show()

################################# 此处脑内容分割完成 #################################

ret, thresh = cv.threshold(img_gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# 将脑内容中的‘点’单独提取出来
# 不加这5行代码的效果很有意思

# thresh = cv.bitwise_not(thresh, thresh)
# ret, markers2 = cv.connectedComponents(thresh)
# markers2 = markers2 + 1
# thresh[markers2 == 2] = 0
# plt.imshow(markers2); plt.show()

# ------ 我现在甚至认为到这一步就已经结束了，分水岭似乎不太适合细小的血管的分割 ------

print(thresh.shape)
plt.subplot(2,3,1)
plt.title('gray')
plt.imshow(thresh,'gray')

# noise removal
kernel = np.ones((3,3),np.uint8)        # 3x3矩阵，所有元素为1
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 1)      # OPEN形态学变换
# opening = cv.bitwise_not(opening, opening)
# 这种操作不得行，应该提取脑CT中的'点'

plt.subplot(2,3,2)
plt.title('morphologyEx,(3,3),2')
plt.imshow(opening)

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)     
# 计算图像中每一个非零点距离离自己最近的[0]的距离，即计算填水速度
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,cv.THRESH_BINARY)
print(sure_fg.dtype)

# Finding unknown region
sure_fg = np.uint8(sure_fg)     # 数据类型转化
print(sure_fg.dtype)
unknown = cv.subtract(sure_bg,sure_fg)      # 计算数组间的元素差

plt.subplot(2,3,3)
plt.title('dist_transform')
plt.imshow(dist_transform)

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
plt.imshow(markers)

markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]

plt.subplot(2,3,5); plt.title('img'); plt.imshow(img)

plt.subplot(2,3,6); plt.title('markers'); plt.imshow(markers)

plt.show()

# 初步感觉分水岭的效果不是太理想



# #
# 2019.7.13
# Wang ChongZhi: 先采用均值滤波直接处理整张图片,(模糊脑内容)？
# 

img1 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0061.tif"
img2 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0073.tif"
img3 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0085.tif"
img4 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0097.tif"
img5 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0109.tif"
img6 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0121.tif"
img7 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0133.tif"
img8 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0145.tif"
img9 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0157.tif"
img9 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0169.tif"

# img = cv.imread('D:/auxiliaryPlane/project/Python/medImgSegment/imgData/'+img5,0) #直接读为灰度图像
'''
blur1 = cv.blur(img,(3,5))  #模板大小3*5
ret, thresh1 = cv.threshold(blur1,254,255,cv.THRESH_BINARY)
blur2 = cv.blur(img,(5,5))
ret, thresh2 = cv.threshold(blur2,254,255,cv.THRESH_BINARY)
blur3 = cv.blur(img,(9,9))
ret, thresh3 = cv.threshold(blur3,254,255,cv.THRESH_BINARY)
plt.subplot(2,2,1),plt.imshow(img,'gray')   #默认彩色，另一种彩色bgr
plt.subplot(2,2,2),plt.imshow(thresh1)
plt.subplot(2,2,3),plt.imshow(thresh2)
plt.subplot(2,2,4),plt.imshow(thresh3)
plt.show()
'''

# 以这种方式不靠谱，3x3的滤波，即使9次膨胀也不可能使得区域封闭
'''
kernel1 = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh3, cv.MORPH_OPEN, kernel1, iterations = 3)
plt.subplot(2,2,1),plt.imshow(thresh1)
sure_bg1 = cv.dilate(opening,kernel1,iterations=5)
plt.subplot(2,2,2),plt.imshow(sure_bg1)
sure_bg1 = cv.dilate(opening,kernel1,iterations=7)
plt.subplot(2,2,3),plt.imshow(sure_bg1)
sure_bg1 = cv.dilate(opening,kernel1,iterations=9)
plt.subplot(2,2,4),plt.imshow(sure_bg1)
plt.show()
'''

# --------------------------- 使用轮廓检测中的凸包函数 ---------------------------
'''
ret, thresh = cv.threshold(img,254,255,cv.THRESH_BINARY)

kernel1 = np.ones((5,5),np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel1, iterations = 3)
plt.subplot(1,2,1), plt.imshow(opening)
sure_bg = cv.dilate(opening,kernel1,iterations=5)
plt.subplot(1,2,2), plt.imshow(sure_bg)
plt.show()

# 图片轮廓
image, contours, hierarchy = cv.findContours(sure_bg, 2, 1)
cnt = contours[0]
# 寻找凸包并绘制凸包(轮廓)
hull = cv.convexHull(cnt)
print(hull)

length = len(hull)
for i in range(len(hull)):
    cv.line(opening, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (255,0,0), 2)

# 显示图片
cv.imshow('line', opening)
cv.waitKey()
'''

# --------------------- Hough Circle Transform 哈夫圆变换 ---------------------
# 这个不知道是跑不出来还是怎么的实在是太慢了，而且估计跑出来也就是找圆（眼珠）
'''
img = cv.imread('D:/auxiliaryPlane/project/Python/medImgSegment/houghcircles2.jpg',0)
# img = cv.imread('opencv-logo-white.png',0)
img = cv.medianBlur(img,5)
# plt.imshow(img), plt.show()
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv.imshow('detected circles',cimg)
cv.waitKey(0)
cv.destroyAllWindows()
'''


# -------------- Interactive Foreground Extraction using GrabCut Algorithm 允许交互式的手工标记 --------------

# img = cv.imread('messi5.jpg')

'''
img = cv.imread('D:/auxiliaryPlane/project/Python/medImgSegment/imgData/'+img5)

img_gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
ret, thresh = cv.threshold(img_gray,254,255,cv.THRESH_BINARY)
kernel1 = np.ones((5,5),np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel1, iterations = 3)

mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (50,50,450,290)
cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()

# newmask is the mask image I manually labelled
# newmask = cv.imread('newmask.png',0)

newmask = cv.imread('D:/auxiliaryPlane/project/Python/medImgSegment/imgData/'+img5, 0)
plt.subplot(1,2,1), plt.imshow(newmask)
plt.subplot(1,2,2), plt.imshow(mask)
plt.show()

# wherever it is marked white (sure foreground), change mask=1
# wherever it is marked black (sure background), change mask=0
mask[newmask == 0] = 0
mask[newmask == 255] = 1
mask, bgdModel, fgdModel = cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()
'''


