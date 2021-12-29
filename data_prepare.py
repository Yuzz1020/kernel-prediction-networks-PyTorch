#导入相关模块
from torch.utils.data import DataLoader,Dataset
from skimage import io,transform
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2xyz
import inspect
from utils.training_util import read_config
from data_generation.data_utils import *
import torch.nn.functional as F
class Random_Horizontal_Flip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        if np.random.rand() < self.p:
            return torch.flip(tensor, dims=[-1])
        return tensor

class Random_Vertical_Flip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, tensor):
        if np.random.rand() < self.p:
            return torch.flip(tensor, dims=[-2])
        return tensor
        
class Customized_dataset(Dataset): #继承Dataset
    def __init__(self, config_file, train_dir, config_spec=None, color= False, train=True,  transform=None, magnitude=1): #__init__是初始化该类的一些基础参数
        if config_spec is None:
            config_spec = self._configspec_path()
        config = read_config(config_file, config_spec)
        self.dataset_config = config['dataset_configs']
        self.dataset_dir = train_dir

        self.train_dir = train_dir   #文件目录
        self.transform = transform #变换
        self.input = os.listdir(self.train_dir)#目录里的所有文件

        self.burst_size = self.dataset_config['burst_length']
        self.patch_size = self.dataset_config['patch_size']

        self.upscale = self.dataset_config['down_sample']
        self.big_jitter = self.dataset_config['big_jitter']
        self.small_jitter = self.dataset_config['small_jitter']
        # 对应下采样之前图像的最大偏移量
        self.jitter_upscale = self.big_jitter * self.upscale
        # 对应下采样之前的图像的patch尺寸
        self.size_upscale = self.patch_size * self.upscale + 2 * self.jitter_upscale
        # 产生大jitter和小jitter之间的delta  在下采样之前的尺度上
        self.delta_upscale = (self.big_jitter - self.small_jitter) * self.upscale
        # 对应到原图的patch的尺寸
        self.patch_size_upscale = self.patch_size * self.upscale

        self.magnitude = magnitude
        
        self.color = color
        self.train = train

        self.vertical_flip = Random_Vertical_Flip(p=0.5)
        self.horizontal_flip = Random_Horizontal_Flip(p=0.5)

    def __len__(self):#返回整个数据集的大小
        return len(self.input)

    @staticmethod
    def _configspec_path():
        current_dir = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe()))
        )
        return os.path.join(current_dir,
                            'dataset_specs/data_configspec.conf')

    @staticmethod
    def crop_random(tensor, patch_size):
        return random_crop(tensor, 1, patch_size)[0]


    def __getitem__(self,index):#根据索引index返回dataset[index]
        # print(index)
        image = Image.open(os.path.join(self.dataset_dir, self.input[index]))
        if not self.color:
            image = ImageOps.grayscale(image)
        else: 
            raise Exception("Only gray scale is supported")
        # 先转换为Tensor进行degamma
        image = transforms.ToTensor()(image)


        image_crop = self.crop_random(image, self.size_upscale)
        # N*H*W  对应于较小jitter下
        image_crop_small = image_crop[:, self.delta_upscale:-self.delta_upscale,
                                        self.delta_upscale:-self.delta_upscale]

        # 进一步进行random_crop所需的transform

        # burst中的第一个不做偏移  后期作为target
        # output shape: N*3*H*W
        img_burst = []
        for i in range(self.burst_size+1):
            if i == 0:
                img_burst.append(
                    image_crop[:, self.jitter_upscale:-self.jitter_upscale, self.jitter_upscale:-self.jitter_upscale]
                )
            else:
                if np.random.binomial(1, min(1.0, np.random.poisson(lam=1.5) / self.burst_size)) == 0:
                    img_burst.append(
                        self.crop_random(
                            image_crop_small, self.patch_size_upscale
                        )
                    )
                else:  #big
                    img_burst.append(
                        self.crop_random(image_crop, self.patch_size_upscale)
                    )
        image_burst = torch.stack(img_burst, dim=0)
        image_burst = F.adaptive_avg_pool2d(image_burst, (self.patch_size, self.patch_size))

        # label为patch中burst的第一个
        if not self.color:
            # image_burst = 0.2989*image_burst[:, 0, ...] + 0.5870 * image_burst[:, 1, ...] + 0.1140*image_burst[:, 2, ...]
            image_burst = torch.clamp(image_burst, 0.0, 1.0)
        else:
            raise Exception("Only gray scale is supported")


        if self.train:
            # data augment
            image_burst = self.horizontal_flip(image_burst)
            image_burst = self.vertical_flip(image_burst)

        label = image_burst[0, ...]
        img = image_burst[1:, ...]

        #TODO: potentially add degamma and white level

        # generate the binary frames burst
        img = torch.poisson(img*self.magnitude)
        img = (img > 0).float()
        return img.squeeze(),label.squeeze() #返回该样本

