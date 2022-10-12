
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


def select_model(name):
    '''
    this function returns the models specified.
    '''
    
    if name == 'r18':    
        #resnet18
        resnet18 = models.resnet18()
        resnet18.conv1 = nn.Conv2d(3,64,kernel_size=(3,3),padding=(1,1), bias=False)
        resnet18.fc = nn.Linear(512,10)
        return resnet18

    elif name == 'r50':
        #resnet50
        resnet50 = models.resnet50()
        resnet50.conv1 = nn.Conv2d(3,64,kernel_size=(3,3),padding=(1,1), bias=False)
        resnet50.fc = nn.Linear(2048,10)
        return resnet50
    
    elif name == 'r101':
        #resnet101
        resnet101 = models.resnet101()
        resnet101.conv1 = nn.Conv2d(3,64,kernel_size=(3,3),padding=(1,1), bias=False)
        resnet101.fc = nn.Linear(2048,10)
        return resnet101

    elif name == 'dnet121':
        #densenet121
        densenet121 = models.densenet121(weights=None)
        densenet121.features.conv0 = nn.Conv2d(3,64,kernel_size=(3,3),padding=(1,1), bias=False)
        densenet121.classifier = nn.Linear(1024,10)
        return densenet121

    elif name == 'vgg16':
        #vgg16
        vgg16 = models.vgg16()
        vgg16.classifier[-1] = nn.Linear(4096, 10)
        return vgg16

    elif name == 'vgg19':
        #vgg19
        vgg19 = models.vgg19()
        vgg19.classifier[-1] = nn.Linear(4096, 10)
        return vgg19

    elif name == 'alexnet':
        # alexnet 
        alexnet = models.AlexNet()
        alexnet.features[0] = nn.Conv2d(3,64,kernel_size=(3,3),padding=(1,1), bias=False)
        alexnet.classifier[-1] = nn.Linear(4096,10)
        return alexnet

    elif name == 'mnetv2':
        # mobilenetv2
        mobilenetV2 = models.mobilenet_v2()
        mobilenetV2.classifier[-1] = nn.Linear(1280,10)
        return mobilenetV2

    elif name == 'mnetv3':
        #mobilenetv3
        mobilenetV3 = models.mobilenet_v3_large()
        mobilenetV3.classifier[-1] = nn.Linear(1280,10)
        return mobilenetV3

    elif name == 'cnext_tiny':
        #convnet_tiny
        convnext_tiny = models.convnext_tiny()
        convnext_tiny.classifier[-1] = nn.Linear(768,10)           
        return convnet_tiny

    elif name == 'cnext_base':
        #convnext_base
        convnext_base = models.convnext_base()
        convnext_base.classifier[-1] = nn.Linear(1024,10)
        return convnext_base

    elif name == 'swin_base':
        #swinbase 
        swin_base = models.swin_b()
        swin_base.head = nn.Linear(1024,10)
        return swin_base

    elif name == 'effnetv2':
        #efficient net
        efficientnet_v2 = models.efficientnet_v2_s()
        efficientnet_v2.classifier[-1] = nn.Linear(1280,10)
        return efficientnet_v2

    else:
        print('please select the correct model!')