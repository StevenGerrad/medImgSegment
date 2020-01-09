
###################################################################################
# 
#       2019.7.10
#       建立血管三维图
#       --------------------- 
#       作者：clovermini 
#       来源：CSDN 
#       原文：https://blog.csdn.net/baidu_33122327/article/details/86541585 
#       版权声明：本文为博主原创文章，转载请附上博文链接！
# 
###################################################################################

import os
import h5py
import numpy as np
import cv2

img_path = "D:/MINE_FILE/dataSet/CTA/CTA1/"
out_h5_path = 'D:/auxiliaryPlane/project/Python/medImgSegment/Documents/blood.h5'

dataset = np.zeros((178, 1208, 1261), np.float)

# 遍历指定地址下的所有图片
cnt_num = 0
img_list = sorted(os.listdir(img_path))
os.chdir(img_path)

for img in img_list:
    if img.endswith(".tif"):
        print(img)
        gray_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        print (gray_img.shape)
        dataset[cnt_num, :, :] = gray_img
        cnt_num += 1

with h5py.File(out_h5_path, 'w') as f:
    f['data'] = dataset  # 将数据写入文件的主键data下面
