
import numpy as np
import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import MRFusion
import random

class BasicBlock_large(nn.Module):
    # BasicBlock places the stride for downsampling at 3x3 convolution for nn.conv3d
    # according to Bottleneck in torchvision.resnet 
    # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    def __init__(self,
                 mode: str,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int=1,
                 padding: int=1,
                 output_padding: int=1,
                 use_batchnorm: bool=True,
                 leaky: bool=True):
        super(BasicBlock_large, self).__init__()

        if mode == 'Encoder':
            self._conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        elif mode == 'Decoder':
            self._conv1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        else:
            print ("Wrong mode, please enter 'Encoder' or 'Decoder'.")
            return
        self._conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self._conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if leaky:
            self._relu = nn.LeakyReLU(0.2) 
        else:    
            self._relu = nn.ReLU(inplace=True)
        self._use_batchnorm = use_batchnorm
        if self._use_batchnorm:
            self._bn_1 = nn.BatchNorm3d(out_channels)
            self._bn_2 = nn.BatchNorm3d(out_channels)
            self._bn_3 = nn.BatchNorm3d(out_channels)
        else:
            self._conv1 = nn.utils.weight_norm(self._conv1, name='weight')
            self._conv2 = nn.utils.weight_norm(self._conv2, name='weight')
            self._conv3 = nn.utils.weight_norm(self._conv3, name='weight')
           
    def forward(self, x):
        out = None
        identity = None

        if self._use_batchnorm:
            out = self._conv1(x)
            out = self._bn_1(out)
            out = self._relu(out)

            identity = out
            out = self._conv2(out)
            out = self._bn_2(out)
            out = self._relu(out)
            out = self._conv3(out)
            out = self._bn_3(out)
        
        else:
            out = self._conv1(x)
            out = self._relu(out)

            identity = out
            out = self._conv2(out)
            out = self._relu(out)
            out = self._conv2(out)

        out += identity
        out = self._relu(out)

        return out

class BasicBlock_MR(nn.Module):
    # BasicBlock places the stride for downsampling at 3x3 convolution for nn.conv3d
    # according to Bottleneck in torchvision.resnet 
    # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

    def __init__(self,
                 mode: str,
                 in_channels: int,
                 out_channels: int,
                 kernel_size=[1,3],
                 stride=[2,2],
                 padding=[0,1],
                 output_padding: int=1,
                 use_batchnorm: bool=True,
                 leaky: bool=False):
        super(BasicBlock_MR, self).__init__()
        M=2
        if mode == 'Encoder':
            self._conv1 = MRFusion(in_channels, out_channels,kernel=kernel_size, stride=stride, padding=padding)
        elif mode == 'Decoder':
            self._conv1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        else:
            print ("Wrong mode, please enter 'Encoder' or 'Decoder'.")
            return
        self._conv2 = nn.Conv3d(out_channels*len(kernel_size), out_channels, kernel_size=3, stride=1, padding=1)
        # self._conv3 = MRFusion(out_channels, out_channels,kernel=[1,3], stride=[1,1],padding=[0,1])
        self._conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if leaky:
            self._relu = nn.LeakyReLU(0.2) 
        else:    
            self._relu = nn.ReLU(inplace=True)
        self._use_batchnorm = use_batchnorm
        if self._use_batchnorm:
            self._bn_1 = nn.BatchNorm3d(out_channels*len(kernel_size))
            self._bn_2 = nn.BatchNorm3d(out_channels)
            self._bn_3 = nn.BatchNorm3d(out_channels)
        else:
            self._conv1 = nn.utils.weight_norm(self._conv1, name='weight')
            self._conv2 = nn.utils.weight_norm(self._conv2, name='weight')
            self._conv3 = nn.utils.weight_norm(self._conv3, name='weight')
           
    def forward(self, x):
        out = None
        identity = None

        if self._use_batchnorm:
            out, mrf_list = self._conv1(x)
            out = self._bn_1(out)
            out = self._relu(out)

            out = self._conv2(out)
            out = self._bn_2(out)
            out = self._relu(out)

            identity = out
            out = self._conv3(out)
            out = self._bn_3(out)
        
        else:
            out = self._conv1(x)
            out = self._relu(out)
            out = self._conv2(out)
            out = self._relu(out)

            identity = out    
            out = self._conv2(out)

        out += identity
        out = self._relu(out)

        return out, mrf_list

class PSLN(nn.Module):
    def __init__(self, use_batchnorm, device, channel_num):
        super(PSLN, self).__init__()
        self._device = device
        self._channel_num = channel_num
        self._priors_path = 'priors'

        self._encoder_input_1_16 = BasicBlock_MR('Encoder', 1, int(self._channel_num / 2), kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#16
        self._encoder_input_2_8 = BasicBlock_MR('Encoder', int(self._channel_num / 2), int(self._channel_num),kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#8
        self._encoder_input_3_4 = BasicBlock_MR('Encoder', int(self._channel_num), int(self._channel_num*2), kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#4
        self._encoder_input_4_2 = BasicBlock_MR('Encoder', int(self._channel_num*2), int(self._channel_num*2), kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#2

        self._encoder_priors_1_16 = BasicBlock_MR('Encoder', 1, int(self._channel_num / 2), kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#16
        self._encoder_priors_2_8 = BasicBlock_MR('Encoder', int(self._channel_num / 2), int(self._channel_num), kernel_size=[7,5,3], stride=[2,2,2], padding=[3,2,1], use_batchnorm=use_batchnorm)#8
        self._encoder_priors_3_4 = BasicBlock_MR('Encoder', int(self._channel_num), int(self._channel_num*2), kernel_size=[5,3], stride=[2,2], padding=[2,1], use_batchnorm=use_batchnorm)#4
        self._encoder_priors_4_2 = BasicBlock_MR('Encoder', int(self._channel_num*2), int(self._channel_num*2), kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#2

        self._decoder_2 = BasicBlock_large('Decoder', self._channel_num*4, int(self._channel_num*2), 3, 2, use_batchnorm=use_batchnorm)
        self._decoder_4 = BasicBlock_large('Decoder', int(self._channel_num*6), int(self._channel_num), 3, 2, use_batchnorm=use_batchnorm) 
        self._decoder_8 = BasicBlock_large('Decoder', int(self._channel_num*3), int(self._channel_num), 4, 4, output_padding=2, use_batchnorm=use_batchnorm)
        self._conv_last = nn.Conv3d(int(self._channel_num), 1, 3, 1, 1)
   
        self._softmax = nn.Softmax(1)
        self._relu = nn.ReLU(inplace=False)

        priors = self._get_data()

        priors = torch.from_numpy(np.array(priors)).unsqueeze(1)
        self._codebook = priors.float().to(self._device)
        self._sgm = nn.Sigmoid()
    def _get_data(self):
        """
        This function reads all the shape priors for training classes generated by using ShapeNet GT models for training samples
        """
        priors = []
        for proir_file in glob.iglob(os.path.join(self._priors_path, "*_voxel.npy")):
                prior = np.load(proir_file)
                priors.append(prior)
        return priors

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        instance_features_16,_= self._encoder_input_1_16(inputs) #b c/4 16 16 16
        instance_features_8,_ = self._encoder_input_2_8(instance_features_16)#b 64 8 8 8
        instance_features_4,_ = self._encoder_input_3_4(instance_features_8)#b 64 4 4 4
        instance_features_2,_ = self._encoder_input_4_2(instance_features_4)#b 64 2 2 2 
        
        priors = torch.clamp(self._codebook, min=0, max=1)
        prior_features_16,_ = self._encoder_priors_1_16(priors)
        prior_features_8, plist_8 = self._encoder_priors_2_8(prior_features_16)
        prior_features_4, plist_4 = self._encoder_priors_3_4(prior_features_8)
        prior_features_2, plist_2 = self._encoder_priors_4_2(prior_features_4)#b 2c 2 2 2 

        ## for res 2
        patch_features_2 = torch.flatten(instance_features_2, start_dim=2).transpose(1, 2) #b 8 64
        patch_features_2 = torch.flatten(patch_features_2, start_dim=0, end_dim=1)#b*8 64
        prior_patch_features_2 = torch.flatten(plist_2[0], start_dim=2).transpose(1, 2) #112 8 64
        prior_patch_features_2 = torch.flatten(prior_patch_features_2, start_dim=0, end_dim=1)#112*8 64
        patch_features_2 = patch_features_2 / (self._channel_num/4)
        learned_weights_2 = torch.matmul(patch_features_2, prior_patch_features_2.transpose(1, 0))#b*8 65408
        learned_weights_2 = self._softmax(learned_weights_2)#b 8c
        shape_features_2 = torch.matmul(learned_weights_2, prior_patch_features_2) #b*8 64
        shape_features_2 = shape_features_2.view(batch_size, 8, int(self._channel_num*2)).transpose(1, 2).view(batch_size, int(self._channel_num*2), 2, 2, 2)
        shape_features_2_4 = self._decoder_2(torch.cat([shape_features_2,instance_features_2], dim=1))#b 64 4 4 4

        patch_features_4 = torch.flatten(instance_features_4, start_dim=2).transpose(1, 2) #b 64 64
        patch_features_4 = torch.flatten(patch_features_4, start_dim=0, end_dim=1)#b*64 64
        prior_patch_features_4_1 = torch.flatten(plist_4[0], start_dim=2).transpose(1, 2) #112 64 64
        prior_patch_features_4_1 = torch.flatten(prior_patch_features_4_1, start_dim=0, end_dim=1)#112*64 64
        
        prior_patch_features_4_2 = torch.flatten(plist_4[1], start_dim=2).transpose(1, 2) #112 64 64
        prior_patch_features_4_2 = torch.flatten(prior_patch_features_4_2, start_dim=0, end_dim=1)#112*64 64
        prior_patch_features_4 = torch.cat([prior_patch_features_4_1,prior_patch_features_4_2], dim=0)#14336 64
        patch_features_4 = patch_features_4 / (self._channel_num/4)
        learned_weights_4 = torch.matmul(patch_features_4, prior_patch_features_4.transpose(1, 0))#b*64 65408
        
        learned_weights_4 = self._softmax(learned_weights_4)#b 8c
        shape_features_4 = torch.matmul(learned_weights_4, prior_patch_features_4) #b*64 64
        shape_features_4 = shape_features_4.view(batch_size, 64, int(self._channel_num*2)).transpose(1, 2).view(batch_size, int(self._channel_num*2), 4, 4, 4)
        shape_features_4_8 = self._decoder_4(torch.cat([shape_features_2_4, shape_features_4,instance_features_4], dim=1))#b 64 8 8 8
        
        patch_features_8 = torch.flatten(instance_features_8, start_dim=2).transpose(1, 2) #b 512 64
        patch_features_8 = torch.flatten(patch_features_8, start_dim=0, end_dim=1)#b*512 64
        prior_patch_features_8_1 = torch.flatten(plist_8[0], start_dim=2).transpose(1, 2) #112 8 64
        prior_patch_features_8_1 = torch.flatten(prior_patch_features_8_1, start_dim=0, end_dim=1)#112*8 64
        
        prior_patch_features_8_2 = torch.flatten(plist_8[1], start_dim=2).transpose(1, 2) #112 64 64
        prior_patch_features_8_2 = torch.flatten(prior_patch_features_8_2, start_dim=0, end_dim=1)#112*64 64

        prior_patch_features_8_3 = torch.flatten(plist_8[2], start_dim=2).transpose(1, 2) #112 512 64
        prior_patch_features_8_3 = torch.flatten(prior_patch_features_8_3, start_dim=0, end_dim=1)#112*512 64
        
        prior_patch_features_8 = torch.cat([prior_patch_features_8_1,prior_patch_features_8_2,prior_patch_features_8_3], dim=0)#14336 64
        patch_features_8 = patch_features_8 / (self._channel_num/4)
        learned_weights_8 = torch.matmul(patch_features_8, prior_patch_features_8.transpose(1, 0))#b*512 65408
        learned_weights_8 = self._softmax(learned_weights_8)#b 8c
        shape_features_8 = torch.matmul(learned_weights_8, prior_patch_features_8) #b*512 64
        shape_features_8 = shape_features_8.view(batch_size, 512, int(self._channel_num)).transpose(1, 2).view(batch_size, int(self._channel_num), 8, 8, 8)
        final_shape = self._decoder_8(torch.cat([shape_features_4_8, shape_features_8,instance_features_8], dim=1))#b 64 32 32 32
        final_shape = self._conv_last(final_shape)
        final_shape = self._sgm(final_shape)
        return final_shape, instance_features_8, instance_features_4, instance_features_2

class PSLN_CaSR(nn.Module):
    def __init__(self, use_batchnorm, device, channel_num, ctg=None):
        super(PSLN_CaSR, self).__init__()
        """
        """
        self._device = device
        self._channel_num = channel_num
        # prior path
        if ctg != 'all':
            self._priors_path = 'part_fusion_priors112/'+ctg
        else:
            self._priors_path = 'priors'
        self._encoder_input_1_16 = BasicBlock_MR('Encoder', 1, int(self._channel_num / 2), kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#16
        self._encoder_input_2_8 = BasicBlock_MR('Encoder', int(self._channel_num / 2), int(self._channel_num),kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#8
        self._encoder_input_3_4 = BasicBlock_MR('Encoder', int(self._channel_num), int(self._channel_num*2), kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#4
        self._encoder_input_4_2 = BasicBlock_MR('Encoder', int(self._channel_num*2), int(self._channel_num*2), kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#2
        
        self._encoder_priors_1_16 = BasicBlock_MR('Encoder', 1, int(self._channel_num / 2), kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#16
        self._encoder_priors_2_8 = BasicBlock_MR('Encoder', int(self._channel_num / 2), int(self._channel_num), kernel_size=[7,5,3], stride=[2,2,2], padding=[3,2,1], use_batchnorm=use_batchnorm)#8
        self._encoder_priors_3_4 = BasicBlock_MR('Encoder', int(self._channel_num), int(self._channel_num*2), kernel_size=[5,3], stride=[2,2], padding=[2,1], use_batchnorm=use_batchnorm)#4
        self._encoder_priors_4_2 = BasicBlock_MR('Encoder', int(self._channel_num*2), int(self._channel_num*2), kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#2

        self._decoder_2 = BasicBlock_large('Decoder', self._channel_num*4, int(self._channel_num*2), 3, 2, use_batchnorm=use_batchnorm)
        self._decoder_4 = BasicBlock_large('Decoder', int(self._channel_num*6), int(self._channel_num), 3, 2, use_batchnorm=use_batchnorm) 
        self._decoder_8 = BasicBlock_large('Decoder', int(self._channel_num*3), int(self._channel_num), 4, 4, output_padding=2, use_batchnorm=use_batchnorm)
        self._conv_last = nn.Conv3d(int(self._channel_num), 1, 3, 1, 1)
   
        self._softmax = nn.Softmax(1)
        self._relu = nn.ReLU(inplace=False)

        priors = self._get_data()
        priors = torch.from_numpy(np.array(priors)).unsqueeze(1)
        self._codebook = None
        self._codebook = priors.float().to(self._device)
        self._sgm = nn.Sigmoid()

    def _get_data(self):
        priors = []
        for proir_file in glob.iglob(os.path.join(self._priors_path, "*_voxel.npy")):
            with open(proir_file, 'rb') as data:
                prior = np.load(data)
                priors.append(prior)           
        return priors
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        instance_features_16,_= self._encoder_input_1_16(inputs) #b c/4 16 16 16
        instance_features_8,_ = self._encoder_input_2_8(instance_features_16)#b 64 8 8 8
        instance_features_4,_ = self._encoder_input_3_4(instance_features_8)#b 64 4 4 4
        instance_features_2,_ = self._encoder_input_4_2(instance_features_4)#b 64 2 2 2 
        
        priors = torch.clamp(self._codebook, min=0, max=1)
        prior_features_16,_ = self._encoder_priors_1_16(priors)
        prior_features_8, plist_8 = self._encoder_priors_2_8(prior_features_16)
        prior_features_4, plist_4 = self._encoder_priors_3_4(prior_features_8)
        prior_features_2, plist_2 = self._encoder_priors_4_2(prior_features_4)#b 2c 2 2 2 

        ## for res 2
        patch_features_2 = torch.flatten(instance_features_2, start_dim=2).transpose(1, 2) #b 8 64
        patch_features_2 = torch.flatten(patch_features_2, start_dim=0, end_dim=1)#b*8 64
        prior_patch_features_2 = torch.flatten(plist_2[0], start_dim=2).transpose(1, 2) #112 8 64
        prior_patch_features_2 = torch.flatten(prior_patch_features_2, start_dim=0, end_dim=1)#112*8 64
        patch_features_2 = patch_features_2 / (self._channel_num/4)
        learned_weights_2 = torch.matmul(patch_features_2, prior_patch_features_2.transpose(1, 0))#b*8 65408
        learned_weights_2 = self._softmax(learned_weights_2)#b 8c
        shape_features_2 = torch.matmul(learned_weights_2, prior_patch_features_2) #b*8 64
        shape_features_2 = shape_features_2.view(batch_size, 8, int(self._channel_num*2)).transpose(1, 2).view(batch_size, int(self._channel_num*2), 2, 2, 2)
        shape_features_2_4 = self._decoder_2(torch.cat([shape_features_2,instance_features_2], dim=1))#b 64 4 4 4
        
        patch_features_4 = torch.flatten(instance_features_4, start_dim=2).transpose(1, 2) #b 64 64
        patch_features_4 = torch.flatten(patch_features_4, start_dim=0, end_dim=1)#b*64 64
        prior_patch_features_4_1 = torch.flatten(plist_4[0], start_dim=2).transpose(1, 2) #112 64 64
        prior_patch_features_4_1 = torch.flatten(prior_patch_features_4_1, start_dim=0, end_dim=1)#112*64 64
        
        prior_patch_features_4_2 = torch.flatten(plist_4[1], start_dim=2).transpose(1, 2) #112 64 64
        prior_patch_features_4_2 = torch.flatten(prior_patch_features_4_2, start_dim=0, end_dim=1)#112*64 64
        prior_patch_features_4 = torch.cat([prior_patch_features_4_1,prior_patch_features_4_2], dim=0)#14336 64
        patch_features_4 = patch_features_4 / (self._channel_num/4)
        learned_weights_4 = torch.matmul(patch_features_4, prior_patch_features_4.transpose(1, 0))#b*64 65408
        
        learned_weights_4 = self._softmax(learned_weights_4)#b 8c
        shape_features_4 = torch.matmul(learned_weights_4, prior_patch_features_4) #b*64 64
        shape_features_4 = shape_features_4.view(batch_size, 64, int(self._channel_num*2)).transpose(1, 2).view(batch_size, int(self._channel_num*2), 4, 4, 4)
        shape_features_4_8 = self._decoder_4(torch.cat([shape_features_2_4, shape_features_4,instance_features_4], dim=1))#b 64 8 8 8
        
        patch_features_8 = torch.flatten(instance_features_8, start_dim=2).transpose(1, 2) #b 512 64
        patch_features_8 = torch.flatten(patch_features_8, start_dim=0, end_dim=1)#b*512 64
        prior_patch_features_8_1 = torch.flatten(plist_8[0], start_dim=2).transpose(1, 2) #112 8 64
        prior_patch_features_8_1 = torch.flatten(prior_patch_features_8_1, start_dim=0, end_dim=1)#112*8 64
        
        prior_patch_features_8_2 = torch.flatten(plist_8[1], start_dim=2).transpose(1, 2) #112 64 64
        prior_patch_features_8_2 = torch.flatten(prior_patch_features_8_2, start_dim=0, end_dim=1)#112*64 64

        prior_patch_features_8_3 = torch.flatten(plist_8[2], start_dim=2).transpose(1, 2) #112 512 64
        prior_patch_features_8_3 = torch.flatten(prior_patch_features_8_3, start_dim=0, end_dim=1)#112*512 64
        
        prior_patch_features_8 = torch.cat([prior_patch_features_8_1,prior_patch_features_8_2,prior_patch_features_8_3], dim=0)#14336 64
        patch_features_8 = patch_features_8 / (self._channel_num/4)
        learned_weights_8 = torch.matmul(patch_features_8, prior_patch_features_8.transpose(1, 0))#b*512 65408
        learned_weights_8 = self._softmax(learned_weights_8)#b 8c
        shape_features_8 = torch.matmul(learned_weights_8, prior_patch_features_8) #b*512 64
        shape_features_8 = shape_features_8.view(batch_size, 512, int(self._channel_num)).transpose(1, 2).view(batch_size, int(self._channel_num), 8, 8, 8)
        final_shape = self._decoder_8(torch.cat([shape_features_4_8, shape_features_8,instance_features_8], dim=1))#b 64 32 32 32
        final_shape = self._conv_last(final_shape)
        final_shape = self._sgm(final_shape)
        return final_shape


class PSLN_D_CaSR(nn.Module):
    def __init__(self, use_batchnorm, device, channel_num, ctg=None):
        super(PSLN_D_CaSR, self).__init__()
        self._device = device
        self._channel_num = channel_num
        if ctg != 'all':
            self._priors_path = 'part_fusion_priors112/'+ctg
        else:
            self._priors_path = 'priors'
        self._encoder_input_1_16 = BasicBlock_MR('Encoder', 1, int(self._channel_num / 2), kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#16
        self._encoder_input_2_8 = BasicBlock_MR('Encoder', int(self._channel_num / 2), int(self._channel_num),kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#8
        self._encoder_input_3_4 = BasicBlock_MR('Encoder', int(self._channel_num), int(self._channel_num*2), kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#4
        self._encoder_input_4_2 = BasicBlock_MR('Encoder', int(self._channel_num*2), int(self._channel_num*2), kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#2
        
        self._encoder_priors_1_16 = BasicBlock_MR('Encoder', 1, int(self._channel_num / 2), kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#16
        self._encoder_priors_2_8 = BasicBlock_MR('Encoder', int(self._channel_num / 2), int(self._channel_num), kernel_size=[7,5,3], stride=[2,2,2], padding=[3,2,1], use_batchnorm=use_batchnorm)#8
        self._encoder_priors_3_4 = BasicBlock_MR('Encoder', int(self._channel_num), int(self._channel_num*2), kernel_size=[5,3], stride=[2,2], padding=[2,1], use_batchnorm=use_batchnorm)#4
        self._encoder_priors_4_2 = BasicBlock_MR('Encoder', int(self._channel_num*2), int(self._channel_num*2), kernel_size=[3], stride=[2], padding=[1], use_batchnorm=use_batchnorm)#2

        self._decoder_2 = BasicBlock_large('Decoder', self._channel_num*4, int(self._channel_num*2), 3, 2, use_batchnorm=use_batchnorm)
        self._decoder_4 = BasicBlock_large('Decoder', int(self._channel_num*6), int(self._channel_num), 3, 2, use_batchnorm=use_batchnorm) 
        self._decoder_8 = BasicBlock_large('Decoder', int(self._channel_num*3), int(self._channel_num), 4, 4, output_padding=2, use_batchnorm=use_batchnorm)
        self._conv_last = nn.Conv3d(int(self._channel_num), 1, 3, 1, 1)
   
        self._softmax = nn.Softmax(1)
        self._relu = nn.ReLU(inplace=False)

        priors = self._get_data()
        priors = torch.from_numpy(np.array(priors)).unsqueeze(1)
        self._codebook = priors.float().to(self._device)
        self._sgm = nn.Sigmoid()

    def _get_data(self):
        """
        This function reads all the shape priors for training classes generated by using ShapeNet GT models for training samples
        """
        priors = []
        for proir_file in glob.iglob(os.path.join(self._priors_path, "*.npy")):
            with open(proir_file, 'rb') as data:
                prior = np.load(data)
                priors.append(prior)
            
        return priors
    def forward(self, inputs):
        batch_size = inputs.shape[0]
        instance_features_16,_= self._encoder_input_1_16(inputs) #b c/4 16 16 16
        instance_features_8,_ = self._encoder_input_2_8(instance_features_16)#b 64 8 8 8
        instance_features_4,_ = self._encoder_input_3_4(instance_features_8)#b 64 4 4 4
        instance_features_2,_ = self._encoder_input_4_2(instance_features_4)#b 64 2 2 2 
        
        priors = torch.clamp(self._codebook, min=0, max=1)
       
        prior_features_8, plist_8 = self._encoder_priors_2_8(prior_features_16)
        prior_features_4, plist_4 = self._encoder_priors_3_4(prior_features_8)
        prior_features_2, plist_2 = self._encoder_priors_4_2(prior_features_4)#b 2c 2 2 2 

        ## for res 2
        patch_features_2 = torch.flatten(instance_features_2, start_dim=2).transpose(1, 2) #b 8 64
        
        patch_features_2 = torch.flatten(patch_features_2, start_dim=0, end_dim=1)#b*8 64
        
        prior_patch_features_2 = torch.flatten(plist_2[0], start_dim=2).transpose(1, 2) #112 8 64
        
        prior_patch_features_2 = torch.flatten(prior_patch_features_2, start_dim=0, end_dim=1)#112*8 64
        
        patch_features_2 = patch_features_2 / (self._channel_num/4)
        learned_weights_2 = torch.matmul(patch_features_2, prior_patch_features_2.transpose(1, 0))#b*8 65408
        
        learned_weights_2 = self._softmax(learned_weights_2)#b 8c
        
        shape_features_2 = torch.matmul(learned_weights_2, prior_patch_features_2) #b*8 64
        
        shape_features_2 = shape_features_2.view(batch_size, 8, int(self._channel_num*2)).transpose(1, 2).view(batch_size, int(self._channel_num*2), 2, 2, 2)
        shape_features_2_4 = self._decoder_2(torch.cat([shape_features_2,instance_features_2], dim=1))#b 64 4 4 4
        
        patch_features_4 = torch.flatten(instance_features_4, start_dim=2).transpose(1, 2) #b 64 64
       
        patch_features_4 = torch.flatten(patch_features_4, start_dim=0, end_dim=1)#b*64 64
       
        prior_patch_features_4_1 = torch.flatten(plist_4[0], start_dim=2).transpose(1, 2) #112 64 64
        
        prior_patch_features_4_1 = torch.flatten(prior_patch_features_4_1, start_dim=0, end_dim=1)#112*64 64
        
        prior_patch_features_4_2 = torch.flatten(plist_4[1], start_dim=2).transpose(1, 2) #112 64 64
        
        prior_patch_features_4_2 = torch.flatten(prior_patch_features_4_2, start_dim=0, end_dim=1)#112*64 64
        prior_patch_features_4 = torch.cat([prior_patch_features_4_1,prior_patch_features_4_2], dim=0)#14336 64
        
        patch_features_4 = patch_features_4 / (self._channel_num/4)
        learned_weights_4 = torch.matmul(patch_features_4, prior_patch_features_4.transpose(1, 0))#b*64 65408
        
        learned_weights_4 = self._softmax(learned_weights_4)#b 8c
       
        shape_features_4 = torch.matmul(learned_weights_4, prior_patch_features_4) #b*64 64
        
        shape_features_4 = shape_features_4.view(batch_size, 64, int(self._channel_num*2)).transpose(1, 2).view(batch_size, int(self._channel_num*2), 4, 4, 4)
        shape_features_4_8 = self._decoder_4(torch.cat([shape_features_2_4, shape_features_4,instance_features_4], dim=1))#b 64 8 8 8
        
        patch_features_8 = torch.flatten(instance_features_8, start_dim=2).transpose(1, 2) #b 512 64
       
        patch_features_8 = torch.flatten(patch_features_8, start_dim=0, end_dim=1)#b*512 64
        
        prior_patch_features_8_1 = torch.flatten(plist_8[0], start_dim=2).transpose(1, 2) #112 8 64
       
        prior_patch_features_8_1 = torch.flatten(prior_patch_features_8_1, start_dim=0, end_dim=1)#112*8 64
        
        prior_patch_features_8_2 = torch.flatten(plist_8[1], start_dim=2).transpose(1, 2) #112 64 64
        
        prior_patch_features_8_2 = torch.flatten(prior_patch_features_8_2, start_dim=0, end_dim=1)#112*64 64

        prior_patch_features_8_3 = torch.flatten(plist_8[2], start_dim=2).transpose(1, 2) #112 512 64
        
        prior_patch_features_8_3 = torch.flatten(prior_patch_features_8_3, start_dim=0, end_dim=1)#112*512 64
        
        prior_patch_features_8 = torch.cat([prior_patch_features_8_1,prior_patch_features_8_2,prior_patch_features_8_3], dim=0)#14336 64
        patch_features_8 = patch_features_8 / (self._channel_num/4)
        learned_weights_8 = torch.matmul(patch_features_8, prior_patch_features_8.transpose(1, 0))#b*512 65408
        
        learned_weights_8 = self._softmax(learned_weights_8)#b 8c
        
        shape_features_8 = torch.matmul(learned_weights_8, prior_patch_features_8) #b*512 64
        shape_features_8 = shape_features_8.view(batch_size, 512, int(self._channel_num)).transpose(1, 2).view(batch_size, int(self._channel_num), 8, 8, 8)
        offset = self._decoder_8(torch.cat([shape_features_4_8, shape_features_8,instance_features_8], dim=1))#b 64 32 32 32
        offset = self._conv_last(offset)
   
        coarse_shape_invsgm = -torch.log((1/(inputs+1e-8))-1)
        coarse_shape_invsgm += offset 
        final_shape = self._sgm(coarse_shape_invsgm)
        return final_shape




