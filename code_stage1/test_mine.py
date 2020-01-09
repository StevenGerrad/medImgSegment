
###################################################################################
#
#       2019.7.7
#       医学脑血管图像分割
#       mine
# 
###################################################################################

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import string

img1 = "DE #PP DE_CarotidAngio 1.0 Q30f 3 A_80kV.0061.tif"
global imgPtr
imgPtr = 0

def showImg(image):
    global imgPtr
    cnt = str(imgPtr)
    cv.namedWindow(cnt+"-img", cv.WINDOW_NORMAL)
    # cv.resizeWindow(imgPtr+"-img", 512, 512)
    cv.imshow(cnt+'-img',image)
    imgPtr = imgPtr + 1
    # cv.waitKey(0)

def threshold_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)  #灰度化
    #直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    print("threshold value %s"%ret)
    showImg(binary)
    return binary

src = cv.imread('D:/auxiliaryPlane/project/Python/medImgSegment/imgData/'+img1)
showImg(src)

# 1. 灰度化
threshold_demo(src)

# 2. 噪声处理
#   效果不明显啊，他这个怎么去判断是否是噪声呢，还是看是否前后图'血管'的连通性，因而先不考虑'噪声'
imgDenoise = cv.fastNlMeansDenoising(src,None,3,7,21)

# 3. 降噪后灰度化
threshold_demo(imgDenoise)



cv.waitKey(0)



