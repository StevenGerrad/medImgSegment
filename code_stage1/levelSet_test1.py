
###################################################################################
# 
#       2019.7.9
#       levelset分割
#       --------------------- 
#       作者：GlassySky0816 
#       来源：CSDN 
#       原文：https://blog.csdn.net/qq_38784098/article/details/82144106 
#   
###################################################################################

'''
import cv2
from pylab import*
from matplotlib import pyplot as plt

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


def inHead_seg(img):
    # plt.subplot(2,3,1); plt.title('original'); plt.imshow(img)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # plt.subplot(2,3,2); plt.title('img_gray'); plt.imshow(img_gray)
    ret, thresh = cv2.threshold(img_gray,254,255,cv2.THRESH_BINARY)
    # plt.subplot(2,3,3); plt.title('thresh'); plt.imshow(thresh)
    kernel = np.ones((5,5),np.uint8) 
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
    print(opening.shape)
    # plt.subplot(2,3,4); plt.title('opening'); plt.imshow(opening)
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    sure_bg = cv2.bitwise_not(sure_bg,sure_bg)
    # plt.subplot(2,3,5); plt.title('sure_bg'); plt.imshow(sure_bg)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_bg)
    markers = labels
    print(markers.shape)
    markers = markers+1
    img[markers != 3] = 255
    # plt.subplot(2,3,6); plt.title('img'); plt.imshow(img)
    # plt.show()
    return img

######################################## content ###########################################

Image = cv2.imread('D:/auxiliaryPlane/project/Python/medImgSegment/imgData/'+img2, 1)
# Image = cv2.imread('D:/auxiliaryPlane/project/Python/medImgSegment/'+img2, 1)

Image = inHead_seg(Image)     # 这个函数把脑内容分割出来 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# 去除图像中细小的白噪声,需要考虑一下
# Image = cv2.pyrMeanShiftFiltering(Image,10,30)        #这个模糊没什么软用啊,处理完噪声更多了

kernel = np.ones((3,3),np.uint8)
Image = cv2.morphologyEx(Image,cv2.MORPH_OPEN,kernel, iterations = 2)

image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
img = np.array(image, dtype=np.float64)  # 构造img,读入到np的array中,并转化浮点类型
 
# 初始水平集函数
IniLSF = np.ones((img.shape[0], img.shape[1]), img.dtype)
IniLSF[300:320, 300:320] = -1           # ??????????????????????????????????????????????/
IniLSF = -IniLSF

# 画初始轮廓
Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
plt.figure(1), plt.imshow(Image), plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.contour(IniLSF, [0], color='b', linewidth=2)  # 画LSF=0处的等高线
plt.draw(), plt.show(block=False)

 
def mat_math(intput, str):
    output = intput
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if str == "atan":       # 反正切弧度值
                output[i, j] = math.atan(intput[i, j])
            if str == "sqrt":
                output[i, j] = math.sqrt(intput[i, j])
    return output
 
# CV函数
def CV(LSF, img, mu, nu, epison, step):
    Drc = (epison / math.pi) / (epison*epison + LSF*LSF)
    Hea = 0.5*(1 + (2 / math.pi)*mat_math(LSF/epison, "atan"))
    Iy, Ix = np.gradient(LSF)
    s = mat_math(Ix*Ix+Iy*Iy, "sqrt")
    Nx = Ix / (s+0.000001)
    Ny = Iy / (s+0.000001)  # 似乎是得到了正切余切
    Mxx, Nxx = np.gradient(Nx)
    Nyy, Myy = np.gradient(Ny)  # 计算数组梯度
    cur = Nxx + Nyy
    Length = nu*Drc*cur
 
    Lap = cv2.Laplacian(LSF, -1)
    Penalty = mu*(Lap - cur)
 
    s1 = Hea*img
    s2 = (1-Hea)*img
    s3 = 1-Hea
    C1 = s1.sum() / Hea.sum()
    C2 = s2.sum() / s3.sum()
    CVterm = Drc*(-1 * (img - C1)*(img - C1) + 1 * (img - C2)*(img - C2))
 
    LSF = LSF + step*(Length + Penalty + CVterm)
    # plt.imshow(s, cmap ='gray'),plt.show()
    return LSF
 
# 模型参数
mu = 1
nu = 0.003 * 255 * 255
# num = 20    # 设置迭代次数
num = 20
epison = 1
step = 0.1
LSF = IniLSF
for i in range(1, num):
    LSF = CV(LSF, img, mu, nu, epison, step)  # 迭代
    print (mu, nu, epison, step)
    if i % 1 == 0:  # 显示分割轮廓
        plt.imshow(Image), plt.xticks([]), plt.yticks([])
        plt.contour(LSF, [0], colors='r', linewidth=2)
        plt.draw(), plt.show(block=False), plt.pause(0.01)

plt.show()

# 初步观察结果可能的确有细小噪声需要消除

plt.title('LSF')
plt.imshow(LSF)     # 这个LSF的值需要好好琢磨一下
plt.show()

# res = np.zeros((LSF.shape[0], LSF.shape[1]), np.uint8)
# res[LSF > 0] = 1, res[LSF <= 0] = 0       # 这回这么写怎么就没效果了呢

ret, res = cv2.threshold(LSF,0,255,cv2.THRESH_BINARY)
res = np.uint8(res)     # ATTENTION,需要进行数据类型转化

ret, markers = cv2.connectedComponents(res)
markers = markers + 1
plt.title('markers'),plt.imshow(markers),plt.show()

res[markers == 2] = 0
plt.title('res'),plt.imshow(res),plt.show()

'''





###################################################################################
# 
#       2019.7.25
#       levelset分割
#       --------------
#       分割函数封装
#   
###################################################################################


import glob
import os
import cv2
from pylab import*
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

#获取指定目录下的所有图片
# print glob.glob(r"E:/Picture/*/*.jpg")
# print glob.glob(r'../*.py') #相对路径

def inHead_seg(img):
    copyImg = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros([h+2, w+2],np.uint8)   #mask必须行和列都加2，且必须为uint8单通道阵列
    cv2.floodFill(copyImg, mask, (int(h/2), int(w/2)), (5, 5, 5), (55, 55, 55), (55, 55 ,55), cv2.FLOODFILL_FIXED_RANGE)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(img_gray,254,255,cv2.THRESH_BINARY)
    thresh[:,:] = 0
    copyImg_gray = cv2.cvtColor(copyImg,cv2.COLOR_RGB2GRAY)
    thresh[ copyImg_gray==5 ] = 255
    kernel = np.ones((5,5),np.uint8) 
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
    sure_bg = cv2.dilate(opening,kernel,iterations=5)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_bg)
    markers = labels
    markers = markers+1
    img[markers != 2] = 255
    return img

def blurrHand(Image):
    kernel = np.ones((3,3),np.uint8)
    Image = cv2.morphologyEx(Image,cv2.MORPH_OPEN,kernel, iterations = 2)
    return Image


def mat_math(intput, str, img):
    output = intput
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if str == "atan":       # 反正切弧度值
                output[i, j] = math.atan(intput[i, j])
            if str == "sqrt":
                output[i, j] = math.sqrt(intput[i, j])
    return output
 
# CV函数
def CV(LSF, img, mu, nu, epison, step):
    Drc = (epison / math.pi) / (epison*epison + LSF*LSF)
    Hea = 0.5*(1 + (2 / math.pi)*mat_math(LSF/epison, "atan", img))
    Iy, Ix = np.gradient(LSF)
    s = mat_math(Ix*Ix+Iy*Iy, "sqrt", img)
    Nx = Ix / (s+0.000001)
    Ny = Iy / (s+0.000001)  # 似乎是得到了正切余切
    Mxx, Nxx = np.gradient(Nx)
    Nyy, Myy = np.gradient(Ny)  # 计算数组梯度
    cur = Nxx + Nyy
    Length = nu*Drc*cur
 
    Lap = cv2.Laplacian(LSF, -1)
    Penalty = mu*(Lap - cur)
 
    s1 = Hea*img
    s2 = (1-Hea)*img
    s3 = 1-Hea
    C1 = s1.sum() / Hea.sum()
    C2 = s2.sum() / s3.sum()
    CVterm = Drc*(-1 * (img - C1)*(img - C1) + 1 * (img - C2)*(img - C2))
 
    LSF = LSF + step*(Length + Penalty + CVterm)
    # plt.imshow(s, cmap ='gray'),plt.show()
    return LSF

def levelSet(Image):
    image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    img = np.array(image, dtype=np.float64)  # 构造img,读入到np的array中,并转化浮点类型
    # 初始水平集函数
    IniLSF = np.ones((img.shape[0], img.shape[1]), img.dtype)
    IniLSF[300:320, 300:320] = -1          
    IniLSF = -IniLSF
    # 画初始轮廓
    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
    # plt.figure(1), plt.imshow(Image), plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.contour(IniLSF, [0], color='b', linewidth=2)  # 画LSF=0处的等高线
    # plt.draw(), plt.show(block=False)
    # 模型参数
    mu = 1
    nu = 0.003 * 255 * 255
    # num = 20    # 设置迭代次数
    num = 20
    epison = 1
    step = 0.1
    LSF = IniLSF
    for i in range(1, num):
        LSF = CV(LSF, img, mu, nu, epison, step)  # 迭代
        print ("num:",i,mu, nu, epison, step)
        # if i % 1 == 0:  # 显示分割轮廓
            # plt.imshow(Image), plt.xticks([]), plt.yticks([])
            # plt.contour(LSF, [0], colors='r', linewidth=2)
            # plt.draw(), plt.show(block=False), plt.pause(0.01)

    # plt.title('LSF'),plt.imshow(LSF),plt.show()
    ret, res = cv2.threshold(LSF,0,255,cv2.THRESH_BINARY)
    res = np.uint8(res)     # ATTENTION,需要进行数据类型转化
    ret, markers = cv2.connectedComponents(res)
    markers = markers + 1
    # plt.title('markers'),plt.imshow(markers),plt.show()
    res[markers == 2] = 0
    plt.show()
    # print (res)
    # plt.title('res'),plt.imshow(res),plt.show()
    return res

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

Image = cv2.imread('D:/auxiliaryPlane/project/Python/medImgSegment/imgData/'+img2, 1)
Image = inHead_seg(Image)
Image = blurrHand(Image)
levelSet(Image)