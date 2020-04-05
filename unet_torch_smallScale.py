
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
from torchvision import transforms as tfs

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import math

# im_aug1 = tfs.Compose([
#     tfs.Resize(200),
#     tfs.RandomHorizontalFlip(),
#     tfs.RandomCrop(128),
#     tfs.ToTensor()
# ])

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
        # self.conv4 = DoubleConv(256, 512)
        # self.pool4 = nn.MaxPool2d(2)
        # self.conv5 = DoubleConv(512, 1024)

        # 逆卷积，也可以使用上采样
        # self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        # self.conv6 = DoubleConv(1024, 512)
        # self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        # self.conv7 = DoubleConv(512, 256)
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
        # p3 = self.pool3(c3)
        # c4 = self.conv4(p3)
        # p4 = self.pool4(c4)
        # c5 = self.conv5(p4)

        # up_6 = self.up6(c5)
        # merge6 = torch.cat([up_6, c4], dim=1)
        # c6 = self.conv6(merge6)

        # up_7 = self.up7(c6)
        # merge7 = torch.cat([up_7, c3], dim=1)
        # c7 = self.conv7(merge7)

        up_8 = self.up8(c3)
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
    
    def decode(self, image, min=0.0, max=255.0):
        image[image < min] = min
        image[image > max] = max
        image = image / max
        return image

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
    
    def get_train_data(self,number=50,batch_size=5,channel=1,im_size=(512,512)):
        ''' 
        先试试使用未经处理的图像数据
        注意：channel只能为奇数
        '''
        self.get_image_file()
        self.get_label_file()

        self.batch_size = 5
        self.channel = channel
        
        # 此时数据范围较大, 对某范围内数值进行压缩处理, 其余直接取极值
        l_temp = []
        for i in self.image_data:
            i = self.decode(i, 0.0, 300.0)
            l_temp.append(i)
        self.image_data = l_temp

        train_data = []
        train_label = []
        while len(train_data) < number: 
            l1 = []
            l2 = []
            for i in range(batch_size):
                # 随机从图片序列中选一个channel起始索引
                ind = random.randint(0, len(self.image_data) - channel)
                # channel压缩
                temp = self.image_data[ind]
                # temp = temp[:,:,np.newaxis,np.newaxis]
                temp = temp[np.newaxis,np.newaxis,:,:]
                for i in range(ind + 1, ind + channel):
                    temp1 = self.image_data[i][np.newaxis, np.newaxis, :, :]
                    # torch.cat([temp, temp1], dim=1)
                    temp = np.concatenate((temp, temp1), axis=1)
                l1.append(temp)

                # label只能是三维，channel只能为奇数
                temp = self.label_data[int((ind*2+channel-1)/2)]
                temp = temp[np.newaxis,:,:]
                l2.append(temp)
            
            t1 = l1[0]
            t2 = l2[0]
            for i in range(1,batch_size):
                # torch.cat([t1, l1[i]], dim=3)
                # torch.cat([t2, l2[i]], dim=3)
                t1 = np.concatenate((t1, l1[i]), axis=0)
                t2 = np.concatenate((t2, l2[i]), axis=0)
            # 将numpy转化为torch.tensor
            t1 = torch.from_numpy(t1)
            t1 = torch.tensor(t1, dtype=torch.float32)
            t2 = torch.from_numpy(t2)
            t2 = torch.tensor(t2, dtype=torch.float32)

            # 默认大小为(512,512)的不处理
            if im_size != (512, 512):
                ind_x = round(random.random() * (512 - im_size[0]))
                ind_y = round(random.random() * (512 - im_size[1]))
                ind_xx = torch.LongTensor(list(range(ind_x, ind_x + im_size[0])))
                ind_yy = torch.LongTensor(list(range(ind_y, ind_y + im_size[1])))
                t1 = torch.index_select(t1, 2, ind_xx)
                t1 = torch.index_select(t1, 3, ind_yy)

                t2 = torch.index_select(t2, 1, ind_xx)
                t2 = torch.index_select(t2, 2, ind_yy)
            
            train_data.append(t1)
            train_label.append(t2)
        
        return train_data,train_label
        # return self.image_data, self.label_data

def train_model(net, xx, yy, EPOCH=100, learning_rate=0.05):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # loss_func = torch.nn.CrossEntropyLoss()
    # loss_func = torch.nn.BCELoss()
    loss_func = torch.nn.BCEWithLogitsLoss()

    accuracy = 0
    print('start train...')

    train_loss_history = []
    test_loss_history = []
    for epoch in range(EPOCH):
        step = 0

        # 取 80% 数据做训练集
        ind_test = int(len(xx)*0.8)
        for step, b_x in enumerate(xx[0:ind_test]):
            output = net(b_x)  # cnn output

            # reference: https://www.pytorchtutorial.com/pytorch-u-net/ 但不好用
            # permute such that number of desired segments would be on 4th dimension
            # TODO: 为什么专门把channel放到后面?
            # output = output.permute(0, 2, 3, 1)
            # m = output.shape[0]

            # Resizing the outputs and label to caculate pixel wise softmax loss
            # TODO: width_out = 128, height_out = 128, channel_out = 1
            # output = output.resize(m*128*128, 1)
            # label = yy[step].resize(m*128*128)

            
            loss = loss_func(torch.squeeze(output), torch.squeeze(yy[step]))
            
            # clear gradients for this training step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1 == 0:
                pred = net(b_x)
                # print("\r" + 'Epoch: ' + str(epoch) + ' step: ' + str(step) + '[' +">>>" * int(step / 10) + ']',end=' ')
                print('Epoch:{} step:{}'.format(epoch, step),'loss: %.6f' % loss.data.numpy())
                train_loss_history.append(loss.data.numpy())
                # print('loss: %.4f' % loss.data.numpy(), '| accuracy: %.4f' % accuracy, end=' ')
                # print('loss: %.4f' % loss.data.numpy(), end=' ')
        
        # test
        for step, b_x in enumerate(xx[ind_test:]):
            output = net(b_x)  # cnn output
            # loss = loss_func(output, yy[step])
            loss = loss_func(torch.squeeze(output), torch.squeeze(yy[step]))
            
            # clear gradients for this training step
            optimizer.zero_grad()
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 1 == 0:
                # print("\r" + 'Epoch: ' + str(epoch) + ' step: ' + str(step) + '[' +">>>" * int(step / 10) + ']',end=' ')
                print('Test Epoch:{} step:{}'.format(epoch, step), 'loss: %.6f' % loss.data.numpy(), end='')
                test_loss_history.append(loss.data.numpy())
                print(iou(output, yy[step]))
                # print('loss: %.4f' % loss.data.numpy(), '| accuracy: %.4f' % accuracy, end=' ')
                # print('loss: %.4f' % loss.data.numpy(), end=' ')

def iou(img_true, img_pred):
    img_true = torch.squeeze(img_true)
    img_pred = torch.squeeze(img_pred) 
    img_pred = (img_pred > 0).float()
    i = (img_true * img_pred).sum()
    u = (img_true + img_pred).sum()
    return i / u if u != 0 else uint8

if __name__ == "__main__":
    # 处理源图像与标记数据
    trainDataLoader = trainData()
    # 这个函数目的其实是为了检查一下标记和数据有没有对上
    # trainDataLoader.test_show()

    im_channel = 5
    x, y = trainDataLoader.get_train_data(number=5,batch_size=1,channel=im_channel,im_size=(128,128))
    # 运行unet

    model = Unet(in_ch=im_channel,out_ch=1)
    print(model)
    
    train_model(model,x,y,100,0.05)
    
    print()


