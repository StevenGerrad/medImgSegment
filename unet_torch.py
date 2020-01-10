
################################################################################################
# 
#       2020.1.9
#       -----------
#       pytorch 实现unet https://blog.csdn.net/jiangpeng59/article/details/80189889
# 
################################################################################################



import torch.nn as nn
import torch
from torch import autograd

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

#把常用的2个卷积操作简单封装下
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), #添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        # 逆卷积，也可以使用上采样
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out

class trainData():
    def __init__(self):
        ''' 图像数据与标记数据路径 s1: 各为 150张 '''
        self.image_dir = './imgData/train_data/s1/image/'
        self.label_dir = './imgData/train_data/s1/label/'
        self.image_data = None
        self.label_data = None
        
    def get_image_file(self):
        ''' 图像数据为 512 * 512, 为源数据值, 未作改动(范围在: -500左右 ~ 1000+ ) '''
        file_dir = self.image_dir
        image_data = []

        files = os.listdir(file_dir)
        files.sort(key= lambda x:int(x[:-4]))
        for file in files:
            if os.path.splitext(file)[1] == '.npy':
                img_item = np.load(file_dir + file)
                image_data.append(img_item)
        self.image_data = image_data
        return image_data
    
    def get_label_file(self):
        ''' 标签数据为 512 * 512, 为0/1 '''
        file_dir = self.label_dir
        label_data = []

        files = os.listdir(file_dir)
        files.sort(key= lambda x:int(x[:-4]))
        for file in files:
            if os.path.splitext(file)[1] == '.npy':
                img_item = np.load(file_dir + file)
                label_data.append(img_item)
        self.label_data = label_data
        return label_data
    
    def test_show(self):
        ''' 随机取九章展示 '''
        if self.image_data == None or self.label_data == None:
            self.get_image_file()
            self.get_label_file()
        
        _, figs = plt.subplots(3, 3)
        for i in range(3):
            for j in range(3): 
                index = random.randint(0,len(self.image_data)-1)
                img_item = self.image_data[index]
                label_item = self.label_data[index]
                # 标签处简单处理，显示浅红色
                img_item[img_item < 0] = 0
                img_item[img_item > 255] = 255
                img_item = img_item.astype(np.uint8)
                img_item = cv2.merge([img_item, img_item, img_item])
                img_item[label_item > 0] = img_item[label_item > 0] * 0.6 + (80, 0, 0)

                figs[i][j].imshow(img_item)
        plt.show()
    
    def get_train_data(self):
        ''' 先试试使用未经处理的图像数据 '''
        self.get_image_file()
        self.get_label_file()
        return self.image_data, self.label_data

if __name__ == "__main__":
    # 处理源图像与标记数据
    trainDataLoader = trainData()
    trainDataLoader.test_show()
    
    x,y = trainDataLoader.get_train_data()
    # 运行unet
    # model = Unet()
    
    # 测试准确率



