# -*- coding: utf-8 -*-
'''
@time: 2021 07

@ author: wx
'''
import matplotlib; 
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pywt, os, copy
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal
# from scipy.fft import fft, fftshift
from torchvision import transforms

import time
from PIL import Image

class BoneDataset(Dataset):
    def __init__(self, data_path, csv_path, train=True):
        super(BoneDataset, self).__init__()
        self.bones_df = pd.read_csv(csv_path)
        self.bones_df.iloc[:,1:3] = self.bones_df.iloc[:,1:3].astype(np.float)
        self.data_path = data_path
        self.train = train


    def transformt(self, sample):
        composed_transforms = transforms.Compose([
            transforms.Resize((576, 896)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation((90)),

            #transforms.RandomCrop((288, 288)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms(sample)
    
    def transformv(self, sample):
        composed_transforms = transforms.Compose([
            transforms.Resize((576, 896)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms(sample)



    def __getitem__(self, index):        
        # print(self.file_list[index])
        
        image_path = self.data_path +'/'+ str(int(self.bones_df.iloc[index,0:1])) + '.png'
        img = Image.open(image_path).convert('RGB')
        if self.train == True:
            imgt = self.transformt(img)
        else:
            imgt = self.transformv(img)
        age = np.float(self.bones_df.iloc[index,1:2])
        sex = np.float(self.bones_df.iloc[index,2:3])
        
   
        return imgt, age,sex

    def __len__(self):
        return len(self.bones_df)
    

