
################################################################# IMPORTS ###################################
import argparse
import os 
import time 
import torch
from torchvision import utils
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models

import timm
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import time
import math
import os
import random
##############################################################################################################


torch.manual_seed(42 + 0)
np.random.seed(42 + 0)
random.seed(42 + 0)

############################################################################################################
total_models = ['effnetv2', 'swin_base', 'cnext_base', 'cnext_tiny', 'mnetv3', 'mnetv2', 'alexnet', 'vgg19', 'vgg16', 'dnet121', 'r101', 'r50', 'r18']


def select_model(args):
    '''
    this function returns the models specified.
    '''
    if args.data_config == "cifar10":
        classes = 10
    elif args.data_config == "cifar100":
        classes = 100
    else:
        print('enter correct cofig')

    if args.model == 'r18':    
        #resnet18
        resnet18 = models.resnet18()
        # resnet18.conv1 = nn.Conv2d(3,64,kernel_size=(3,3),padding=(1,1), bias=False)
        resnet18.fc = nn.Linear(512,classes)
        return resnet18

    elif args.model == 'r50':
        #resnet50
        resnet50 = models.resnet50()
        # resnet50.conv1 = nn.Conv2d(3,64,kernel_size=(3,3),padding=(1,1), bias=False)
        resnet50.fc = nn.Linear(2048,classes)
        return resnet50
    
    elif args.model == 'r101':
        #resnet101
        resnet101 = models.resnet101()
        # resnet101.conv1 = nn.Conv2d(3,64,kernel_size=(3,3),padding=(1,1), bias=False)
        resnet101.fc = nn.Linear(2048,classes)
        return resnet101

    elif args.model == 'dnet121':
        #densenet121
        densenet121 = models.densenet121(weights=None)
        # densenet121.features.conv0 = nn.Conv2d(3,64,kernel_size=(3,3),padding=(1,1), bias=False)
        densenet121.classifier = nn.Linear(1024,classes)
        return densenet121

    elif args.model == 'vgg16':
        #vgg16
        vgg16 = models.vgg16()
        vgg16.classifier[-1] = nn.Linear(4096, classes)
        return vgg16

    elif args.model == 'vgg19':
        #vgg19
        vgg19 = models.vgg19()
        vgg19.classifier[-1] = nn.Linear(4096, classes)
        return vgg19

    elif args.model == 'alexnet':
        # alexnet 
        alexnet = models.AlexNet()
        # alexnet.features[0] = nn.Conv2d(3,64,kernel_size=(3,3),padding=(1,1), bias=False)
        alexnet.classifier[-1] = nn.Linear(4096,classes)
        return alexnet

    elif args.model == 'mnetv2':
        # mobilenetv2
        mobilenetV2 = models.mobilenet_v2()
        mobilenetV2.classifier[-1] = nn.Linear(1280,classes)
        return mobilenetV2

    elif args.model == 'mnetv3':
        #mobilenetv3
        mobilenetV3 = models.mobilenet_v3_large()
        mobilenetV3.classifier[-1] = nn.Linear(1280,classes)
        return mobilenetV3

    elif args.model == 'cnext_tiny':
        #convnet_tiny
        convnext_tiny = timm.create_model('convnext_tiny', pretrained=True)
        convnext_tiny.head.fc = nn.Linear(768,10)       
        return convnext_tiny

    elif args.model == 'cnext_base':
        #convnext_base
        convnext = timm.create_model('convnext_base', pretrained=True)
        convnext.head.fc = nn.Linear(1024,10)
        return convnext

    elif args.model == 'swin_base':
        #swinbase 
        swin_base = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        swin_base.head = nn.Linear(1024,classes)
        return swin_base

    elif args.model == 'effnetv2':
        #efficient net
        efficientnet_v2 = models.efficientnet_v2_s()
        efficientnet_v2.classifier[-1] = nn.Linear(1280,classes)
        return efficientnet_v2

    else:
        print('please select the correct model!')
