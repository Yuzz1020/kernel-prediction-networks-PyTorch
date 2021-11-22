#导入相关模块
from torch.utils.data import DataLoader,Dataset
from skimage import io,transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np

class Customized_dataset(Dataset): #继承Dataset
    def __init__(self, train_dir, label_dir, transform=None): #__init__是初始化该类的一些基础参数
        self.train_dir = train_dir   #文件目录
        self.transform = transform #变换
        self.input = os.listdir(self.train_dir)#目录里的所有文件

        self.label_dir = label_dir   #文件目录
        self.label = os.listdir(self.label_dir)#目录里的所有文件
    
    def __len__(self):#返回整个数据集的大小
        return len(self.input)
    
    def __getitem__(self,index):#根据索引index返回dataset[index]
        image_index = self.input[index]#根据索引index获取该图片
        img_path = os.path.join(self.train_dir, image_index)#获取索引为index的图片的路径名
        # img = io.imread(img_path)# 读取该图片
        img = np.load(img_path)

        label_index = self.label[index]#根据索引index获取该图片
        label_path = os.path.join(self.label_dir, label_index)#获取索引为index的图片的路径名
        # img = io.imread(img_path)# 读取该图片
        label = np.load(label_path)
        # img.swapaxes(2,0)
        # label.swapaxes(1,0)
        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        # sample = {'image':img,'label':label}#根据图片和标签创建字典
        # print('sample',sample)
        
        # if self.transform:
        #     sample = self.transform(sample)#对样本进行变换
        return img,label #返回该样本

