
###################################################################################
#
#       2019.7.15
#       医学脑血管图像分割
#       脑内容分割
#       -----------
#       仿交互式分割
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

img = cv.imread('D:/auxiliaryPlane/project/Python/medImgSegment/imgData/'+img5)

# newmask is the mask image I manually labelled
# newmask = cv.imread('newmask.png',0)

img_gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
ret, thresh = cv.threshold(img_gray,254,255,cv.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8) 
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 3)

plt.title('opening'), plt.imshow(opening), plt.show()

img_gray[img_gray == 255] = 254
img_gray[opening == 255] = 255
newmask = img_gray

mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

# wherever it is marked white (sure foreground), change mask=1
# wherever it is marked black (sure background), change mask=0
mask[newmask == 0] = 0
mask[newmask == 255] = 1
mask, bgdModel, fgdModel = cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask[:,:,np.newaxis]
plt.imshow(img),plt.colorbar(),plt.show()

















