#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022-12-23 17:40:14

@author: wangxu
"""


from __future__ import print_function, division
import torch.nn as nn

import torchvision
import torch





#### load model
resnet50 = torchvision.models.resnet50(pretrained=True)
for param in resnet50.parameters():
    param.requires_grad = True

num_ftrs = resnet50.fc.in_features        
resnet50.fc = nn.Linear(num_ftrs, 1024)



class AgePredictor(nn.Module):
    def __init__(self, num_classes=1):
        super(AgePredictor, self).__init__()
        self.resnet = resnet50
        
        self.num_img = 1024
        self.num_gen = 256
        self.gen_fc = nn.Linear(1,self.num_gen)
        self.cat_fc = nn.Linear(self.num_img+self.num_gen,num_classes)    
        self.sigmoid = nn.Sigmoid()
        
        self.relu1 = nn.Tanh()
        self.relu2 = nn.Tanh()
        
    def forward(self,img, gender):
        x = self.resnet(img)
        #x= self.relu1(x)
        y = self.gen_fc(gender.unsqueeze(-1))
        #y = self.relu2(y)
        
        z = torch.cat((x,y),dim = 1)
        z = self.cat_fc(z)
        
        return z


