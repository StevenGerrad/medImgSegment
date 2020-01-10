
################################################################################################
# 
#       2020.1.9
#       -----------
#       标号为 12190000 的文件夹下共三个患者图像
#       s1 数据集为包含 编号75989854的dicom医学图像的患者的全部150张图像与标签
# 
################################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

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
        ''' 随机取九张展示 '''
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
        ''' 
        TODO：先试试使用未经处理的图像数据, 返回对象为包含np矩阵的列表
        '''
        self.get_image_file()
        self.get_label_file()
        return self.image_data, self.label_data

if __name__ == "__main__":
    # 处理源图像与标记数据
    trainDataLoader = trainData()
    trainDataLoader.test_show()
    
    x,y = trainDataLoader.get_train_data()