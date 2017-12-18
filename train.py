#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:01:03 2017

@author: xnat
"""

from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from torch import nn, optim
from torch.autograd import Variable
import scipy.io as sio  
from PIL import Image

n_epochs=300
batch_size=8
base_lr=0.05
wd=0.0001
momentum=0.9

class PDDataset(Dataset):
     
    def __init__(self, transform=None):
        self.NC_list = []
        self.PD_list = []
        for line in open('/home/xnat/PD/dataset/NC.txt'):
            self.NC_list.append(line.strip())  
        for line in open('/home/xnat/PD/dataset/PD.txt'):
            self.PD_list.append(line.strip())  
        self.train_img_path = '/home/xnat/PD/dataset/train'
        self.all_train_img_names = os.listdir(self.train_img_path)  
        self.len = len(self.all_train_img_names) 
        self.transform = transform
        
    def __len__(self):
        return self.len
        
    def __getitem__(self, idx):
        img_name = self.all_train_img_names[idx]
        imageInfo = sio.loadmat(self.train_img_path + '/' + img_name)
        if img_name.split('.')[0] in self.NC_list:
            label = 0
        if img_name.split('.')[0] in self.PD_list:
            label = 1
       
        img = imageInfo['data']
        if self.transform:
            img = self.transform(img)
            
        return img, label
     
    
class ToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img).type(torch.FloatTensor)        
        
#class Normalize(object):
#    def __call__(self, img):
#        return img/img.max()
        
train_transforms = tv.transforms.Compose([
    ToTensor(),
#    Normalize(),
])   

train_dataset = PDDataset(transform=train_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4)

#dataiter = iter(train_dataloader)
#img, label = dataiter.next()

model = tv.models.resnet50(num_classes=2)
#model.fc = nn.Linear(model.fc.in_features, 2)
#model.conv1.in_channels = 121
print(model)

model_cuda = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=wd)
criterion = nn.CrossEntropyLoss(weight=torch.Tensor([5,1]).cuda())

for epoch in range(1, n_epochs + 1):
    
    if float(epoch) / n_epochs > 0.6:
        lr = base_lr * 0.01
    elif float(epoch) / n_epochs > 0.3:
        lr = base_lr * 0.1
    else:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(param_group['lr'])

    for i, (img, label) in enumerate(train_dataloader):     

        model.zero_grad()
        optimizer.zero_grad()
        
        input_var = Variable(img, volatile=False).cuda()
        target_var = Variable(label,  volatile=False, requires_grad=False).cuda()
        output_var = model_cuda(input_var)
        
        loss = criterion(output_var, target_var)
        loss.backward()
        optimizer.step()
        
        if i % 5 == 0 :
            print('%s: (Epoch %d of %d) [%04d/%04d]   Loss:%.5f'
                  % ('Train',epoch, n_epochs, i, len(train_dataloader), loss.data[0]))

    if epoch % 10 == 0:
        torch.save(model.state_dict(), '/home/xnat/PD/weights/PD_resnet'+ '_' + str(epoch))


#%%
#
#NC_img_path = '/home/xnat/PD/dataset/train/NC'
#PD_img_path = '/home/xnat/PD/dataset/train/PD'
#all_NC_img_names = os.listdir(NC_img_path)
#all_PD_img_names = os.listdir(PD_img_path)
#NC_len = len(all_NC_img_names) 
#PD_len = len(all_PD_img_names) 
#
#img_name = all_NC_img_names[0]
#imageInfo = sio.loadmat(NC_img_path + '/' + img_name)
#img = imageInfo['data']
#label = 0
#
#
#    
#    