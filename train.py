####################
# imports
####################
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

from utils.dataset import give_loader
from utils.model2 import select_model
from utils.random import progress_bar

#########################   
# Fix seed
#########################
torch.manual_seed(42 + 0)
np.random.seed(42 + 0)
random.seed(42 + 0)
##########################


######
# argument parser
######
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default='r18', type=str, help='name of the model to train')
parser.add_argument('--data_config', default='cifar10', type=str, help='select from which dataset to work on!')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=200, type=int, help='select the number of epoch you want to run for your dataset')
parser.add_argument('--batch_size', default=128, type=int, help='select the size of the dataset batch')
parser.add_argument('--moment', default=0.9, type=float, help='momentum constant value')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay value')
args = parser.parse_args()

############
# Settings
############
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

##############
# data loader
##############

train_loader, validation_loader, test_loader = give_loader(args.data_config, temp=True)

############
# MODEL
###########
model = select_model(args.model)
model.to(device)
criterion = nn.CrossEntropyLoss()


# optimizers
if args.model == 'effnetv2' or args.model == 'mnetv2' or args.model == 'mnetv3':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,
        momentum=args.moment, weight_decay=args.weight_decay)

elif args.model == 'cnext_base' or args.model == 'cnext_tiny':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

elif args.model == 'swin_base':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.moment, weight_decay=args.weight_decay)


# schedulers
if args.model == 'r18' or 'r50' or 'r101':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150], last_epoch=0-1)
elif args.model == 'dnet121':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[150, 225])
else:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)

############
# TRAIN function
############

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    model.eval()
    NUM_EPOCHS = args.epoch
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validation_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(validation_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint_'+str(args.data_config)):
            os.mkdir('checkpoint_'+str(args.data_config))
        torch.save(state, './checkpoint_'+str(args.data_config)+'/'+args.model+'ckpt.pth')
        best_acc = acc
    
    if not os.path.isdir('checkpoint_for_'+str(args.model)+'_'+str(args.data_config)):
            os.mkdir('checkpoint_for_'+str(args.model)+'_'+str(args.data_config))
    if epoch == math.floor((1/100)*NUM_EPOCHS) or epoch == math.floor((10/100)*NUM_EPOCHS) or epoch == math.floor((30/100)*NUM_EPOCHS) or epoch == math.floor((50/100)*NUM_EPOCHS) or epoch == math.floor((70/100)*NUM_EPOCHS):
        print('saving model at:', epoch)
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        pth = 'checkpoint_for_'+str(args.model)+'_'+str(args.data_config)+'/'+str(args.model)+'_'+str(epoch)
        torch.save(state,pth)



for epoch in range(start_epoch, args.epoch):
    train(epoch)
    test(epoch)
    scheduler.step()
