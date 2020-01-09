
###################################################################################
#
#       2019.7.8
#       医学脑血管图像分割
#       分水岭---部分
#       脑内容分割---全部图片处理
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

#获取指定目录下的所有图片
# print glob.glob(r"E:/Picture/*/*.jpg")
# print glob.glob(r'../*.py') #相对路径

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


WSI_MASK_PATH = 'D:/auxiliaryPlane/project/Python/medImgSegment/imgData'#存放图片的文件夹路径
paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.tif'))
paths.sort()

for ipath in paths:
    img= cv2.imread(ipath)
    img = inHead_seg(img)
    img = Image.fromarray(img, mode='RGB')
    # img.show()
    pre_savename = 'D:/auxiliaryPlane/project/Python/medImgSegment/Documents/'
    line = "inHead"+ipath[-8:-6]+".png"
    print (pre_savename,line)
    img.save(os.path.join(pre_savename,line),'PNG')

'''

###################################################################################
# 
#       2019.7.25
#       --------
#       采用类似画图工具中的'填充'方法的泛洪函数处理
#       2019.7.27
#       对第一次的结果中的异常图像进行分析: 应进行孔洞填充并判断最大面积的连通分量
# 
###################################################################################


import glob
import os
import cv2
from pylab import*
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from itertools import chain

#获取指定目录下的所有图片
# print glob.glob(r"E:/Picture/*/*.jpg")
# print glob.glob(r'../*.py') #相对路径

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

def inHead_seg(img):
    copyImg = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros([h+2, w+2],np.uint8)   #mask必须行和列都加2，且必须为uint8单通道阵列
    # print (img[int(h/2), int(w/2)])
    cv2.floodFill(copyImg, mask, (int(h/2), int(w/2)), (5, 5, 5), (55, 55, 55), (55, 55 ,55), cv2.FLOODFILL_FIXED_RANGE)
    # plt.title('fill_color_demo'),plt.imshow(copyImg),plt.show()
    # plt.subplot(2,3,1); plt.title('original'); plt.imshow(img)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # plt.subplot(2,3,2); plt.title('img_gray'); plt.imshow(img_gray)
    ret, thresh = cv2.threshold(img_gray,254,255,cv2.THRESH_BINARY)
    thresh[:,:] = 0
    copyImg_gray = cv2.cvtColor(copyImg,cv2.COLOR_RGB2GRAY)
    thresh[ copyImg_gray==5 ] = 255
    # plt.subplot(2,3,3); plt.title('thresh'); plt.imshow(thresh)
    kernel = np.ones((5,5),np.uint8)
    thresh = fillHole(thresh)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
    # opening = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 3)
    # opening = fillHole(opening)         # 加入孔洞填充, 得，这个位置还不太好，换到opening前面去

    # print(opening.shape)
    # plt.subplot(2,3,4); plt.title('opening'); plt.imshow(opening)
    sure_bg = cv2.dilate(opening,kernel,iterations=5)
    # sure_bg = cv2.bitwise_not(sure_bg,sure_bg)
    # plt.subplot(2,3,5); plt.title('sure_bg'); plt.imshow(sure_bg)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_bg)
    markers = labels
    # print(markers.shape)
    markers = markers+1
    # plt.subplot(2,3,6); plt.title('img'); plt.imshow(img)
    # plt.show()
    return img,markers,labels

def maxAreaFind(markers):
    markers = list(chain(*markers))
    # print (markers)
    mCnt = np.bincount(markers)		#这东西还只针对一维数据，注意要找的是出现次数第二多的元素
    resFind = np.argmax(mCnt)
    print (mCnt,resFind)
    markers2 = list(filter(lambda x: x != resFind, markers))
    mCnt2 = np.bincount(markers2)
    if(any(mCnt2) == False):
        return resFind
    resFind = np.argmax(mCnt2)
    # print (markers2,resFind)
    # print ('resFind: ',resFind)
    return resFind

# WSI_MASK_PATH = 'D:/auxiliaryPlane/project/Python/medImgSegment/imgData'    #存放图片的文件夹路径
WSI_MASK_PATH = "D:/MINE_FILE/dataSet/CTA/CTA1"
# WSI_MASK_PATH = "medImgSegment/imgData/test1Error"
# WSI_MASK_PATH = "medImgSegment/imgData/testTemp"
paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.tif'))
paths.sort()

for ipath in paths:
    img= cv2.imread(ipath)
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img,markers,labels = inHead_seg(img)
    resFind = maxAreaFind(markers)
    labels = labels + 1
    img[labels == resFind] = img[labels == resFind]*0.6+(0,0,70)
    img_gray[labels != resFind] = 0
    # img[img_gray == 255] = (255,0,0)
    img = Image.fromarray(img, mode='RGB')
    # img.show()
    # pre_savename = 'medImgSegment/Documents/test1.0/'
    pre_savename = 'medImgSegment/Documents/testTemp/'
    line = "inHead_t2"+ipath[-8:-4]+".png"
    print (pre_savename,line)
    img.save(os.path.join(pre_savename,line),'PNG')



###################################################################################
# 
#       2019.7.25
#       --------
#       '填充'泛洪函数+levelSet处理(15times)
#       2019.7.27
#       颅内分割部分结合孔洞填充和最大面积(第二大)判别
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
from itertools import chain

#获取指定目录下的所有图片
# print glob.glob(r"E:/Picture/*/*.jpg")
# print glob.glob(r'../*.py') #相对路径

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
    # img[markers != 2] = 255
    return markers,labels

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
    num = 10
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

def maxAreaFind(markers):
    markers = list(chain(*markers))
    # print (markers)
    mCnt = np.bincount(markers)		#这东西还只针对一维数据，注意要找的是出现次数第二多的元素
    resFind = np.argmax(mCnt)
    # print (mCnt,resFind)
    markers2 = list(filter(lambda x: x != resFind, markers))
    mCnt2 = np.bincount(markers2)
    resFind = np.argmax(mCnt2)
    # print (markers2,resFind)
    # print ('resFind: ',resFind)
    return resFind

# WSI_MASK_PATH = 'D:/auxiliaryPlane/project/Python/medImgSegment/imgData'    #存放图片的文件夹路径
WSI_MASK_PATH = 'D:/auxiliaryPlane/project/Python/medImgSegment/imgData/test2nWe'    #存放图片的文件夹路径
# WSI_MASK_PATH = "D:/MINE_FILE/dataSet/CTA/CTA1"
paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.tif'))
paths.sort()

for ipath in paths:
    img= cv2.imread(ipath)
    copyImg = img.copy()
    markers,labels = inHead_seg(copyImg)
    resFind = maxAreaFind(markers)
    labels = labels + 1
    copyImg[labels != resFind ] = 255
    imag = blurrHand(copyImg)
    res = levelSet(imag)
    img[labels == resFind] = img[labels == resFind]*0.6+(0,0,70)
    img[res == 255] = (255,0,0)
    img = Image.fromarray(img, mode='RGB')
    # img.show()
    # pre_savename = 'D:/auxiliaryPlane/project/Python/medImgSegment/Documents/segWithPoint/'
    pre_savename = 'medImgSegment/Documents/segTest2nWe/'
    line = "inHead_test2nWe"+ipath[-8:-4]+".png"
    print (pre_savename,line)
    img.save(os.path.join(pre_savename,line),'PNG')


'''











