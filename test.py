#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:39:39 2017

@author: xnat
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from torchvision import datasets, models, transforms
from torch import nn, optim
from torch.autograd import Variable
import scipy.io as sio  
from PIL import Image
import csv 
from pandas import read_csv

NC_img_path = '/home/xnat/PD/dataset/valid/NC'
PD_img_path = '/home/xnat/PD/dataset/valid/PD'
all_NC_img_names = os.listdir(NC_img_path)
all_PD_img_names = os.listdir(PD_img_path)
NC_len = len(all_NC_img_names) 
PD_len = len(all_PD_img_names) 

class ToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img).type(torch.FloatTensor)      

test_transforms = tv.transforms.Compose([
    ToTensor(),
#    Normalize(),
])   
        
        
model = tv.models.resnet50(num_classes=2)
model_cuda = model.cuda()
model_cuda.load_state_dict(torch.load('/home/xnat/PD/PD_resnet_10'))


correct = 0
with open("/home/xnat/PD/Predicte.csv","w") as csvfile: 
    writer = csv.writer(csvfile)
    for i in range(NC_len + PD_len):
        if i < NC_len:
            img_name = all_NC_img_names[i]
            imageInfo =  sio.loadmat(NC_img_path + '/' + img_name)
            truth_label = 0
        else:
            img_name = all_PD_img_names[i - NC_len]
            imageInfo =  sio.loadmat(PD_img_path + '/' + img_name)
            truth_label = 1
       
        img = imageInfo['data']
        img = test_transforms(img)
        img.unsqueeze_(0)
        input_var = Variable(img, volatile=True).cuda()
        predict_label = model_cuda(input_var)
        
        row =[]
        row.append(img_name)
        row.extend(predict_label.data[0])
        writer.writerow(row) 
        
        if predict_label.data[0,0] <  predict_label.data[0,1]:
            predict = 1
        else:
            predict = 0
        if predict == truth_label:
            correct = correct +1;
        
acc = correct/(NC_len + PD_len)
print('%d %%' %(acc*100))
