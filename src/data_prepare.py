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
import re
import copy
from data_provider import OnTheFlyDataset, _configspec_path
import argparse
from utils.training_util import MovingAverage, save_checkpoint, load_checkpoint, read_config
from utils.training_util import calculate_psnr, calculate_ssim
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
    def __init__(self, config_file, train_dir, local_window_size, config_spec=None, color= False, train=True,  transform=None, magnitude=1): #__init__是初始化该类的一些基础参数
        if config_spec is None:
            config_spec = self._configspec_path()
        config = read_config(config_file, config_spec)
        self.dataset_config = config['dataset_configs']
        self.local_window_size = int(local_window_size)
        self.dataset_dir = train_dir
        self.train_dir = train_dir   #文件目录
        self.transform = transform #变换
        self.video_list = os.listdir(self.train_dir)#目录里的所有文件
        self.input = []
        self.video_max_size = {}
        self.burst_size = int(self.dataset_config['burst_length'])
        self.patch_size = int(self.dataset_config['patch_size'])
        self.oversample_rate = self.local_window_size
        self.frame_range = (int(self.burst_size) * self.local_window_size) // self.oversample_rate + 1
        if self.frame_range % 2 == 0:
            raise Exception("frame_range should be odd")
        print("frame_range:", self.frame_range)
        for video in self.video_list:
            img_lst = os.listdir(os.path.join(self.train_dir, video))
            self.video_max_size[video] = len(img_lst)
            for img in img_lst:
                frame_id = int(re.findall(r'\d+', img)[0])
                if frame_id < self.frame_range//2 or frame_id >= self.video_max_size[video] - self.frame_range//2:
                    continue
                else:
                    self.input.append(os.path.join(video, img))

        self.upscale = int(self.dataset_config['down_sample'])
        self.big_jitter = int(self.dataset_config['big_jitter'])
        self.small_jitter = int(self.dataset_config['small_jitter'])
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
        if self.color:
            raise Exception("Only gray scale is supported")
        image = Image.open(os.path.join(self.dataset_dir, self.input[index]))
        # image = ImageOps.grayscale(image)
        label = transforms.ToTensor()(image)
        video_name = self.input[index].split('/')[0]
        img_name = self.input[index].split('/')[1]
        frame_id = int(re.findall(r'\d+', img_name)[0])
        frames = range(frame_id - self.frame_range//2, frame_id + self.frame_range//2 + 1)
        image_list= []
        starting_id=frames[0]
        for i in frames:
            if i == starting_id:
                continue
            img_name = video_name + '/' + 'frame' + str(i) + '.jpg'
            image = Image.open(os.path.join(self.dataset_dir, img_name))
            # image = ImageOps.grayscale(image)
            image = transforms.ToTensor()(image)
            image_list.append(image)
        image = torch.cat(image_list, dim=0)


        # over-sample the image between the frames
        image = image.repeat(1,self.oversample_rate,1).reshape(image.shape[0]*self.oversample_rate, image.shape[1],image.shape[2])
        image = F.adaptive_avg_pool2d(image, (self.patch_size, self.patch_size))
        
        image = image.reshape(self.burst_size, -1, image.shape[1], image.shape[2])
        label = image 
        image = torch.poisson(image)
        image = (image > 0).float()
        image = torch.mean(image, dim=1)
        return image.squeeze(),label.squeeze() #返回该样本


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--config_file', dest='config_file', default='kpn_specs/kpn_config.conf', help='path to config file')
    parser.add_argument('--config_spec', dest='config_spec', default='kpn_specs/configspec.conf', help='path to config spec file')
    parser.add_argument('--restart', action='store_true', help='Whether to remove all old files and restart the training process')
    parser.add_argument('--train_dir', type=str, default='/scratch/yz87/test_images/', help='the path to training dataset')
    parser.add_argument('--test_dir', type=str, default='/scratch/yz87/eval_images/', help='the path to evaluation dataset')
    parser.add_argument('--num_workers', '-nw', default=16, type=int, help='number of workers in data loader')
    parser.add_argument('--num_threads', '-nt', default=32, type=int, help='number of threads in data loader')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--mGPU', '-m', action='store_true', help='whether to train on multiple GPUs')
    parser.add_argument('--eval', action='store_true', help='whether to work on the evaluation mode')
    parser.add_argument('--checkpoint', '-ckpt', dest='checkpoint', type=str, default='best',
                        help='the checkpoint to eval')
    parser.add_argument('--in_channel', type=int,           default=50,help='the input channel')
    parser.add_argument("--print_freq", "-pf", default=100, type=int, help="print frequency")
    args = parser.parse_args()
    #
    

    config = read_config(args.config_file, args.config_spec)  
    config['train_dir'] = args.train_dir
    config['test_dir'] = args.test_dir
    config['num_workers'] = args.num_workers
    config['num_threads'] = args.num_threads
    config['cuda'] = args.cuda
    config['mGPU'] = args.mGPU
    config['eval'] = args.eval
    config['checkpoint'] = args.checkpoint
    config['in_channel'] = args.in_channel
    config['print_freq'] = args.print_freq
    config['restart'] = args.restart
    train_config = config['training']
    arch_config = config['architecture']

    batch_size = train_config['batch_size']
    lr = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    decay_step = train_config['decay_steps']
    lr_decay = train_config['lr_decay']

    n_epoch = train_config['num_epochs']
    use_cache = train_config['use_cache']
    trans = transforms.ToPILImage()

    dataset_config = read_config(train_config['dataset_configs'], _configspec_path())['dataset_configs']
    data = Customized_dataset(train_config['dataset_configs'], args.train_dir, train_config['local_window_size'], transform=None, train=False)#初始化类，设置数据集所在路径以及变换
    data_loader = DataLoader(
        data,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=args.num_workers
    )
    dataset_config = read_config(train_config['dataset_configs'], _configspec_path())['dataset_configs']
    for i, (input1,label) in enumerate(data_loader):
        print(input1.shape)
        print(label.shape)
        for img_b in range(train_config["batch_size"]):
            trans(label[img_b].squeeze()).save(os.path.join("./dataset_test", '{}_gt.png'.format(img_b)), quality=100)
            for ti in range(data.burst_size):
                trans(((input1[img_b][ti]).float()).squeeze()).save(os.path.join("./dataset_test", '{}_gt_N{:03d}.png'.format(img_b, ti)), quality=100)
            # for ti in range(4):
                # trans(((input1[img_b][63-ti]).float()).squeeze()).save(os.path.join("./dataset_test", '{}_gt_{}.png'.format(img_b, 63-ti)), quality=100)
            trans(((torch.mean(input1[img_b],dim=0)).float()).squeeze()).save(os.path.join("./dataset_test", '{}_gt_avg.png'.format(img_b)), quality=100)
            input("...")
        input("....")
