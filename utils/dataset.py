
################################################################# IMPORTS ###################################
import argparse
import os
import time
import torch
from torchvision import utils
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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

###############################################   DATA ######################################################


def give_loader(ds):
    '''
    ds : select among cifar10/cifar100
    function which give us the data loader for our experiment
    '''
    print('==> Preparing data..')

    if ds == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_and_valid = torchvision.datasets.CIFAR10(
            root='./datac10', train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(
            root='./datac10', train=False, download=True, transform=transform_test)

    elif ds == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 ([0.2470, 0.2435, 0.2616])),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 ([0.2470, 0.2435, 0.2616])),
        ])

        train_and_valid = torchvision.datasets.CIFAR100(
            root='./datac100', train=True, download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR100(
            root='./datac100', train=False, download=True, transform=transform_test)

    indices = list(range(len(train_and_valid)))  # indices of the dataset
    train_indices, val_indices = train_test_split(
        indices, test_size=0.1, stratify=train_and_valid.targets, random_state=42)

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_sampler.transform = transform_train
    val_sampler.transform = transform_test

    trainloader = torch.utils.data.DataLoader(
        train_and_valid, batch_size=256, shuffle=False, sampler=train_sampler, num_workers=2)

    valloader = torch.utils.data.DataLoader(
        train_and_valid, batch_size=100, shuffle=False, sampler=val_sampler, num_workers=2)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, valloader, testloader

#################################################################################################################
