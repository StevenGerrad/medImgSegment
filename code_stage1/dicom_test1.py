
###################################################################################
#
#       2019.8.30
#       dicom 操作
#       -----------
#       https://blog.csdn.net/zhuang19951231/article/details/79488334
# 
###################################################################################

'''
import os
import pydicom
import numpy
from matplotlib import pyplot

# 用lstFilesDCM作为存放DICOM files的列表
PathDicom = "D:/MINE_FILE/dataSet/CTA/12190000/" #与python文件同一个目录下的文件夹
lstFilesDCM = []
 
for dirName,subdirList,fileList in os.walk(PathDicom):
    for filename in fileList :
        print( filename )
        lstFilesDCM.append(os.path.join(dirName,filename)) # 加入到列表中
		# if ".dcm" in filename.lower():  #判断文件是否为dicom文件

## 将第一张图片作为参考图
RefDs = pydicom.read_file(lstFilesDCM[0])   #读取第一张dicom图片
 
# 建立三维数组
ConstPixelDims = (int(RefDs.Rows),int(RefDs.Columns),len(lstFilesDCM))
 
# 得到spacing值 (mm为单位)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
 
# 三维数据
x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0]) # 0到（第一个维数加一*像素间的间隔），步长为constpixelSpacing
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1]) #
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2]) #
 
ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
 
# 遍历所有的dicom文件，读取图像数据，存放在numpy数组中
for filenameDCM in lstFilesDCM:
    ds = pydicom.read_file(filenameDCM)
    ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

# 轴状面显示
pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.set_cmap(pyplot.gray())
pyplot.pcolormesh(x, y, numpy.flipud(ArrayDicom[:, :, 9])) # 第三个维度表示现在展示的是第几层
pyplot.show()
'''

'''
# 冠状面显示
pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.set_cmap(pyplot.gray())
pyplot.pcolormesh(z, x, numpy.flipud(ArrayDicom[:, 150, :]))
pyplot.show()
'''




###################################################################################
#
#       2019.8.31
#       dicom 图像对比度调整
#       -----------
#       
# 
###################################################################################


###################################################################################
#
#       2019.7.31
#       交互式的调整亮度和对比度
#       -----------
#       https://www.cnblogs.com/lfri/p/10753019.html   
#
###################################################################################

'''
import cv2
import numpy as np

alpha = 0.3
beta = 80
# img_path = "packAirport/image/002.jpg"
img_path = "medImgSegment/img1.tif"
img = cv2.imread(img_path)
img2 = cv2.imread(img_path)

def updateAlpha(x):
    global alpha,img,img2
    alpha = cv2.getTrackbarPos('Alpha','image')
    alpha = alpha * 0.01
    img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))

def updateBeta(x):
    global beta,img,img2
    beta = cv2.getTrackbarPos('Beta','image')
    img = np.uint8(np.clip((alpha * img2 + beta), 0, 255))

# 创建窗口
cv2.namedWindow('image')
cv2.createTrackbar('Alpha','image',0,300,updateAlpha)
cv2.createTrackbar('Beta','image',0,255,updateBeta)
cv2.setTrackbarPos('Alpha','image',100)
cv2.setTrackbarPos('Beta','image',10)
# 设置鼠标事件回调
#cv2.setMouseCallback('image',update)

while(True):
    cv2.imshow('image',img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

'''

# 显示直方图

'''
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import copy
import math

img_path = "medImgSegment/img1.tif"
img_path1 = "bai0.png"
img_path2 = "bai1.png"
img_path3 = "bai2.png"
img_path4 = "bai3.png"

# img = cv.imread('packAirport/image/'+'001.jpg')
img = cv.imread(img_path)
plt.subplot(2,3,1), plt.title('img'), plt.imshow(img)
plt.subplot(2,3,2), plt.hist(img.ravel(),256,[0,256])
plt.subplot(2,3,3)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

gray = img

#伽马变换
# https://blog.csdn.net/yawdd/article/details/80180848/

gamma=copy.deepcopy(gray)
rows=img.shape[0]
cols=img.shape[1]
for i in range(rows):
    for j in range(cols):
        gamma[i][j]=3*pow(gamma[i][j],0.8)

plt.subplot(2,3,4), plt.title('gamma'), plt.imshow(gamma)
plt.subplot(2,3,5), plt.hist(gamma.ravel(),256,[0,256])
plt.subplot(2,3,6)
for i,col in enumerate(color):
    histr = cv.calcHist([gamma],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])

plt.show()
'''

###################################################################################
#
#       2019.8.30
#       dicom 显示
#       -----------
#       https://blog.csdn.net/jaen_tail/article/details/78352446
# 
###################################################################################

'''
import cv2
import numpy
import pydicom
from matplotlib import pyplot as plt
 
# dcm = dicom.read_file("000001.dcm")
dcm = pydicom.read_file("D:/MINE_FILE/dataSet/CTA/12190000/75989104")
dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

slices = []
slices.append(dcm)
img = slices[ int(len(slices)/2) ].image.copy()
ret,img = cv2.threshold(img, 90, 3071, cv2.THRESH_BINARY)
img = numpy.uint8(img)
 
im2, contours, _ = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
mask = numpy.zeros(img.shape, numpy.uint8)
for contour in contours:
    cv2.fillPoly(mask, [contour], 255)
img[(mask > 0)] = 255
 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

img2 = slices[ int(len(slices)/2) ].image.copy()
img2[(img == 0)] = -2000

print ('dicom.shape: ', len(slices))

plt.figure(figsize=(12, 12))
plt.subplot(131)
plt.imshow(slices[int(len(slices) / 2)].image, 'gray')
plt.title('Original')
plt.subplot(132)
plt.imshow(img, 'gray')
plt.title('Mask')
plt.subplot(133)
plt.imshow(img2, 'gray')
plt.title('Result')
plt.show()
'''


###################################################################################
#
#       2019.8.30
#       dicom 显示-- 交互式
#       -----------
#       https://blog.csdn.net/jaen_tail/article/details/78352446
# 
###################################################################################

'''

import os
import cv2
import numpy as np
import pydicom
from matplotlib import pyplot as plt
from itertools import chain

alpha = 0
img_path = "medImgSegment/img1.tif"
dcm = pydicom.read_file("D:/MINE_FILE/dataSet/CTA/12190000/75989104")
dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
print(dir(dcm))
img = dcm.image
print('imgShape: ',img.shape)

PathDicom = "D:/MINE_FILE/dataSet/CTA/12190000/" #与python文件同一个目录下的文件夹
slices = []

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
    mask = np.zeros([h+2, w+2],np.uint8)   # mask必须行和列都加2，且必须为uint8单通道阵列
    # print (img[int(h/2), int(w/2)])
    print('=1=',copyImg.shape)
    copyImg = np.uint8(copyImg)
    cv2.floodFill(copyImg, mask, (int(h/2), int(w/2)), (5, 5, 5), (55, 55, 55), (55, 55 ,55), cv2.FLOODFILL_FIXED_RANGE)
    # plt.title('fill_color_demo'),plt.imshow(copyImg),plt.show()
    # plt.subplot(2,3,1); plt.title('original'); plt.imshow(img)
    img[img > 255] = 255
    img[img < 0] = 0
    # img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_gray = copyImg[:,:,0]
    print('=2=',img_gray.shape)
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

def updateAlpha(x):
    global img, dcm
    alpha = cv2.getTrackbarPos('Alpha','image')
    dcm = pydicom.read_file(slices[ alpha ])
    dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img = dcm.image

    img = cv2.merge((img, img, img))
    img,markers,labels = inHead_seg(img)
    resFind = maxAreaFind(markers)
    labels = labels + 1
    # img[labels == resFind] = img[labels == resFind]*0.6+(0,0,70)
    img[labels != resFind] = 0
    img[img > 255] = 0
    img[img < 100] = 0

for dirName,subdirList,fileList in os.walk(PathDicom):
    for filename in fileList :
        print( filename )
        slices.append(os.path.join(dirName,filename)) # 加入到列表中
		# if ".dcm" in filename.lower():  #判断文件是否为dicom文件

# dcm = pydicom.read_file("D:/MINE_FILE/dataSet/CTA/12190000/75989104")
# dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

# img = slices[ int(len(slices)/2) ].image.copy()
# img[img > 500] = 0
# img[img < -500] = 0

# 创建窗口
cv2.namedWindow('image')
cv2.createTrackbar('Alpha','image',0,160,updateAlpha)
cv2.setTrackbarPos('Alpha','image',0)

while(True):
    cv2.imshow('image',img)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()

'''


###################################################################################
#
#       2019.9.7
#       读取.gz文件
#       -----------
#       https://zhidao.baidu.com/question/1801605038198428147.html
# 
###################################################################################
'''
import gzip
with gzip.open('D:/MINE_FILE/dataSet/CTA/seg12190000.nii.gz', 'rb') as f:
    training_data = f.read()

print (training_data)

'''


###################################################################################
#
#       2019.9.7
#       读取nii文件
#       -----------
#       https://blog.csdn.net/weixin_43330946/article/details/89576759
# 
###################################################################################



import numpy as np
import os                #遍历文件夹
import nibabel as nib    #nii格式一般都会用到这个包
import imageio           #转换成图像
from matplotlib import pyplot as plt
import cv2
 
def nii_to_image(niifile):
    filenames = os.listdir(filepath)  #读取nii文件夹
    slice_trans = []

    for f in filenames:
        #开始读取nii文件
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)                #读取nii
        img_fdata = img.get_fdata()
        fname = f.replace('.nii','')            #去掉nii的后缀名
        # img_f_path = os.path.join(imgfile, fname)
        #创建nii对应的图像的文件夹
        # if not os.path.exists(img_f_path):
            # os.mkdir(img_f_path)                #新建文件夹
 
        #开始转换为图像 (512, 512, 150)
        (x,y,z) = img.shape
        for i in range(0,z):                      #z是图像的序列
            silce = img_fdata[:, :, i]  #选择哪个方向的切片都可以
            
            slice_trans.append(silce)
            # if i < 70 :
            #     continue
            # if np.where(silce!=0)[0].shape[0]!=0:
                # print('this is not a zeros matrix')
            # plt.imshow(silce), plt.show()
            #保存图像
            # imageio.imwrite(os.path.join(img_f_path, '{}.png'.format(i)), silce)
    
    return slice_trans

def get_image_file(file_dir):
    image_data = []
    files = os.listdir(file_dir)
    files.sort(key= lambda x:int(x[:-4]))
    for file in files:
        if os.path.splitext(file)[1] == '.npy':
            img_item = np.load(file_dir + file)
            image_data.append(img_item)
    return image_data
 
if __name__ == '__main__':
    filepath = './imgData/seg12190000'
    label = nii_to_image(filepath)
    file_dir = './Documents/dicom2npy_75989854/'
    image_data = get_image_file(file_dir)

    imgfile = './imgData/label/'
    for i in range(len(image_data)):
        filename_item = imgfile + str(i + 1)
        img_item = image_data[i]

        # 简单对 img 进行处理
        img_item[img_item < 0] = 0
        img_item[img_item > 512] = 512
        # img_item.dtype = np.uint8
        img_item = img_item.astype(np.uint8)
        
        # 不知道为什么只有这样才能整出来
        label_item = label[149-i]
        label_item = np.rot90(label_item, -1)
        label_item = np.flip(label_item, 1)

        # 图片做成三通道查看一下效果
        img_item = cv2.merge([img_item, img_item, img_item])
        img_item[label_item > 0] = img_item[label_item > 0] * 0.6 + (80, 0, 0)
        
        # imageio.imwrite(os.path.join(imgfile, '{}.png'.format(i)), img_item)
        # imageio.imwrite(os.path.join(imgfile, '{}.png'.format(i)), label_item)
        # 保存调试好的label, 反序 + 左右翻转 + 旋转-1*90
        np.save(os.path.join(imgfile, '{}.npy'.format(i)), label_item)
        print(i)




###################################################################################
#
#       2019.9.8
#       dicom 转 jpg
#       -----------
#       
# 
###################################################################################

'''

import SimpleITK as sitk
import numpy as np
import cv2
import os
# import time from PIL
# import Image

count = 1
path = "D:/MINE_FILE/dataSet/CTA/12190000/"
# path = "medImgSegment/imgData/CTA_test1/"
filename = os.listdir(path)
print (filename)

def convert_from_dicom_to_jpg(img,low_window,high_window,save_path):
    lungwin = np.array([low_window*1.,high_window*1.])
    # newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    img[img < 0] = 0
    img[img > 511] = 511
    newimg = (img-0.0)/(512.0)
    newimg = (newimg*255).astype('uint8')
    cv2.imwrite(save_path, newimg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

for i in filename:      # 这样转出的文件数量比源文件多了好多
    document = os.path.join(path,i)
    # outputpath = "./Documents/dicom2jpg"
    outputpath = "./Documents/dicom2npy_test"
    countname = str(count)
    # countfullname = countname + '.jpg'
    countfullname = countname + '.npy'
    output_jpg_path = os.path.join(outputpath, countfullname)
    
    ds_array = sitk.ReadImage(document)
    img_array = sitk.GetArrayFromImage(ds_array)

    shape = img_array.shape     #name.shape
    img_array = np.reshape(img_array, (shape[1], shape[2]))
    high = np.max(img_array)
    low = np.min(img_array)

    # convert_from_dicom_to_jpg(img_array, low, high, output_jpg_path)

    # 12190000 中有三个数据集, 试试是不是后面那一个
    if count > 342:
        outputpath = "./Documents/dicom2npy_test"
        countname = str(count-342)
        # countfullname = countname + '.jpg'
        countfullname = countname + '.npy'
        output_jpg_path = os.path.join(outputpath, countfullname)
        np.save(output_jpg_path, img_array)
    
    print('FINISHED',count-342)

    count = count + 1

'''


###################################################################################
#
#       2019.9.8
#       dicom 显示
#       -----------
#       https://pydicom.github.io/pydicom/stable/auto_examples/image_processing/reslice.html#sphx-glr-auto-examples-image-processing-reslice-py
# 
###################################################################################

'''
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import os

# load the DICOM files
files = []
# path = "D:/MINE_FILE/dataSet/CTA/12190000/"
path = "medImgSegment/imgData/CTA_test1/"
filename = os.listdir(path)

# print('glob: {}'.format(sys.argv[1]))
# for fname in glob.glob(sys.argv[1], recursive=False):
#     print("loading: {}".format(fname))
#     files.append(pydicom.read_file(fname))

for fname in filename:
    document = os.path.join(path,fname)
    print (document)
    # print("loading: {}".format(fname))
    files.append(pydicom.read_file(document))

print("file count: {}".format(len(files)))

# skip files with no SliceLocation (eg scout views)
slices = []
skipcount = 0
for f in files:
    if hasattr(f, 'SliceLocation'):
        slices.append(f)
    else:
        skipcount = skipcount + 1

print("skipped, no SliceLocation: {}".format(skipcount))

# ensure they are in the correct order
slices = sorted(slices, key=lambda s: s.SliceLocation)

# pixel aspects, assuming all slices are the same
ps = slices[0].PixelSpacing
ss = slices[0].SliceThickness
ax_aspect = ps[1]/ps[0]
sag_aspect = ps[1]/ss
cor_aspect = ss/ps[0]

# create 3D array
img_shape = list(slices[0].pixel_array.shape)
img_shape.append(len(slices))
img3d = np.zeros(img_shape)

# fill 3D array with the images from the files
for i, s in enumerate(slices):
    img2d = s.pixel_array
    img3d[:, :, i] = img2d

# plot 3 orthogonal slices
a1 = plt.subplot(2, 2, 1)
plt.imshow(img3d[:, :, img_shape[2]//2])
a1.set_aspect(ax_aspect)

a2 = plt.subplot(2, 2, 2)
plt.imshow(img3d[:, img_shape[1]//2, :])
a2.set_aspect(sag_aspect)

a3 = plt.subplot(2, 2, 3)
plt.imshow(img3d[img_shape[0]//2, :, :].T)
a3.set_aspect(cor_aspect)

plt.show()

'''

