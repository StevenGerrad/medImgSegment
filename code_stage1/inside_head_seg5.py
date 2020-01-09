
###################################################################################
#
#       2019.7.15
#       医学脑血管图像分割
#       脑内容分割
#       -----------
#       https://www.cnblogs.com/FHC1994/p/9033580.html
# 
###################################################################################

#泛洪填充(彩色图像填充)
'''
import cv2 as cv
import numpy as np
import copy
from matplotlib import pyplot as plt

def fill_color_demo(image):
    copyImg = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h+2, w+2],np.uint8)   #mask必须行和列都加2，且必须为uint8单通道阵列
    #为什么要加2可以这么理解：当从0行0列开始泛洪填充扫描时，mask多出来的2可以保证扫描的边界上的像素都会被处理
    # cv.floodFill(copyImg, mask, (int(h/2), int(w/2)), (0, 255, 255), (100, 100, 100), (50, 50 ,50), cv.FLOODFILL_FIXED_RANGE)
    print (image[int(h/2), int(w/2)])
    cv.floodFill(copyImg, mask, (int(h/2), int(w/2)), (0, 255, 255), (60, 60, 60), (60, 60 ,60), cv.FLOODFILL_FIXED_RANGE)
    # cv.namedWindow("fill_color_demo",cv.WINDOW_NORMAL),cv.imshow("fill_color_demo", copyImg)
    plt.title('fill_color_demo'),plt.imshow(copyImg),plt.show()
    cv.imwrite('medImgSegment/Documents/'+'fill_color60-60.jpg', copyImg)

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

src = cv.imread( 'D:/auxiliaryPlane/project/Python/medImgSegment/imgData/'+img5 )
# cv.namedWindow("input_image",cv.WINDOW_NORMAL), cv.imshow('input_image', src)
fill_color_demo(src)
# cv.waitKey(0), cv.destroyAllWindows()

'''


###################################################################################
#
#       2019.7.25
#       医学脑血管图像分割
#       脑内容分割
#       -----------
#       将泛洪填充结合最初的方法进行形态学变换后尝试
# 
###################################################################################

'''
import glob
import os
import cv2
from pylab import*
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img5 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0109.tif"
img = cv2.imread( 'D:/auxiliaryPlane/project/Python/medImgSegment/imgData/'+img5 )

copyImg = img.copy()
h, w = img.shape[:2]
mask = np.zeros([h+2, w+2],np.uint8)   #mask必须行和列都加2，且必须为uint8单通道阵列
print (img[int(h/2), int(w/2)])
cv2.floodFill(copyImg, mask, (int(h/2), int(w/2)), (5, 5, 5), (55, 55, 55), (55, 55 ,55), cv2.FLOODFILL_FIXED_RANGE)
plt.title('fill_color_demo'),plt.imshow(copyImg),plt.show()

plt.subplot(2,3,1); plt.title('original'); plt.imshow(img)

img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
plt.subplot(2,3,2); plt.title('img_gray'); plt.imshow(img_gray)

ret, thresh = cv2.threshold(img_gray,254,255,cv2.THRESH_BINARY)
thresh[:,:] = 0
copyImg_gray = cv2.cvtColor(copyImg,cv2.COLOR_RGB2GRAY)
thresh[ copyImg_gray==5 ] = 255
plt.subplot(2,3,3); plt.title('thresh'); plt.imshow(thresh)

kernel = np.ones((5,5),np.uint8) 
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 3)
print(opening.shape)
plt.subplot(2,3,4); plt.title('opening'); plt.imshow(opening)

sure_bg = cv2.dilate(opening,kernel,iterations=5)
# sure_bg = cv2.bitwise_not(sure_bg,sure_bg)
plt.subplot(2,3,5); plt.title('sure_bg'); plt.imshow(sure_bg)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_bg)
markers = labels
print(markers.shape)
markers = markers+1
img[markers == 2] = img[markers == 2]*0.6+(0,0,70)
plt.subplot(2,3,6); plt.title('img'); plt.imshow(img)

plt.show()

'''


###################################################################################
#
#       2019.7.27
#       医学脑血管图像分割--脑内容分割
#       -----------
#       孔洞填充--这样或许不需要借助形态学变换就可以填充孔洞
#       https://blog.csdn.net/dugudaibo/article/details/84447196
# 
###################################################################################

from itertools import chain

import cv2
import numpy as np
from matplotlib import pyplot as plt


def fillHole(im_in):
	im_floodfill = im_in.copy()
	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = im_in.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)
	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255)
	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)
	# Combine the two images to get the foreground.
	im_out = im_in | im_floodfill_inv

	return im_out

img1 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0055.tif"
img2 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0068.tif"
img3 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0070.tif"
img4 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0071.tif"
img5 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0072.tif"
img6 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0091.tif"
img7 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0092.tif"
img8 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0095.tif"
img9 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0112.tif"
img10= "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0114.tif"
img11= "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0115.tif"

img = cv2.imread( 'D:/auxiliaryPlane/project/Python/medImgSegment/imgData/test1Error/'+img5 )
img = cv2.imread( 'D:/MINE_FILE/dataSet/CTA/CTA1/DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0103.tif')

copyImg = img.copy()
h, w = img.shape[:2]
mask = np.zeros([h+2, w+2],np.uint8)   #mask必须行和列都加2，且必须为uint8单通道阵列
print (img[int(h/2), int(w/2)])
cv2.floodFill(copyImg, mask, (int(h/2), int(w/2)), (5, 5, 5), (55, 55, 55), (55, 55 ,55), cv2.FLOODFILL_FIXED_RANGE)
# plt.title('fill_color_demo'),plt.imshow(copyImg),plt.show()

plt.subplot(2,3,1); plt.title('original'); plt.imshow(img)

img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
plt.subplot(2,3,2); plt.title('img_gray'); plt.imshow(img_gray)

ret, thresh = cv2.threshold(img_gray,254,255,cv2.THRESH_BINARY)
thresh[:,:] = 0
copyImg_gray = cv2.cvtColor(copyImg,cv2.COLOR_RGB2GRAY)
thresh[ copyImg_gray==5 ] = 255
plt.subplot(2,3,3); plt.title('thresh'); plt.imshow(thresh)

kernel = np.ones((5,5),np.uint8) 

thresh = fillHole(thresh)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 3)
# 加入新的孔洞填充+判断连通分量面积大小的手段

# opening = fillHole(opening)

# _,contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# cnt = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]
# print ('连通分量',cnt.shape)
# print (cnt)
# 不使用这个轮廓area了吧，比较麻烦，在markers时使用np的bincnt

print (opening.shape)
plt.subplot(2,3,4); plt.title('opening'); plt.imshow(opening)

sure_bg = cv2.dilate(opening,kernel,iterations=5)
# sure_bg = cv2.bitwise_not(sure_bg,sure_bg)
plt.subplot(2,3,5); plt.title('sure_bg'); plt.imshow(sure_bg)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_bg)
markers = labels

print(markers.shape)
markers = markers + 1
# plt.subplot(2,3,6); plt.title('markers'); plt.imshow(markers)
# print (max(set(markers), key=markers.count))
markers = list(chain(*markers))
# print (markers)
mCnt = np.bincount(markers)		#这狗东西还只针对一维数据，注意要找的是出现次数第二多的元素
resFind = np.argmax(mCnt)
print (mCnt,resFind)
# markers2 = np.mat(filter(lambda x: x != resFind, markers))
markers2 = list(filter(lambda x: x != resFind, markers))

# markers3 = np.mat(filter(lambda x: x != 0, markers2))
mCnt2 = np.bincount(markers2)
resFind = np.argmax(mCnt2)
# print (markers2,resFind)
print ('resFind: ',resFind)
labels = labels + 1

img_gray[labels != resFind] = 0
print ('img_shape: ',img.shape)
img[labels == resFind] = img[labels == resFind]*0.6+(0,0,70)
img[img_gray == 255] = (255,0,0)

plt.subplot(2,3,6); plt.title('img'); plt.imshow(img)

plt.show()
