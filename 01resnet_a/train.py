#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2022年8月9日12:30:07

@author: wangxu

"""
import time
import datetime
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

# from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import BoneDataset
from resnet50 import resnet50,AgePredictor

torch.manual_seed(2022)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_color = [(0),(255)]          

results_path = './results'
exp_name = 'TASNet'

parser = argparse.ArgumentParser(description='feature extarct')
parser.add_argument('--batch_size', type=int, default=16, 
                    help='input batch size for training (default: 20)')
parser.add_argument('--epochs', type=int, default=100, 
                    help='number of epochs to train (default: 100)')
parser.add_argument('--num_workers', type=int, default=0, 
                    help='number of workers to train (default: 0)')
parser.add_argument('--learning_rate', type=float, default=0.00005, 
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.9, 
                    help='momentum (default: 0.9)')
parser.add_argument('--dataset_path', type=str, default='./data/',
                      help='training and validation dataset')
parser.add_argument('--scale', type=int, default=576,help='width')
parser.add_argument('--save_results', type=bool, default=True,help='save result')

args = parser.parse_args()



start_time= time.time()



train_dataset_path = './data_a/train'
val_dataset_path = './data_a/val'
test_dataset_path = './data_a/test'

train_csv_path = './data_a/train.csv'
val_csv_path = './data_a/val.csv'
test_csv_path = './data_a/test.csv'


    # data
train_dataset = BoneDataset(data_path=train_dataset_path, csv_path=train_csv_path, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

val_dataset = BoneDataset(data_path=val_dataset_path, csv_path=val_csv_path, train=False)
val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0)

test_dataset = BoneDataset(data_path=test_dataset_path, csv_path=test_csv_path, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0)
   
print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset),"test_datasize", len(test_dataset))
backbone_path = './backbone/resnet/resnet50-19c8e357.pth'


model_age = AgePredictor(num_classes=1)
model_age = model_age.to(device) 
model_age = nn.DataParallel(model_age)
criterion =  torch.nn.L1Loss() 
optimizer_conv = optim.SGD(model_age.parameters(), lr= args.learning_rate, momentum= args.momentum)  
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=8, gamma=0.1)





def train_model(model_age, criterion, optimizer, scheduler):
    start = time.time()
    res = []
    tq = tqdm(train_dataloader)
    for idx,(imgt, age,sex) in enumerate(tq):
        model_age.train()
        optimizer.zero_grad()
        
        imgt = imgt.to(device).to(torch.float32) 
        age = age.to(device).to(torch.float32)  
        sex = sex.to(device).to(torch.float32) 
        
        pred_age = model_age(imgt,sex)
        # pred_age = pred_age * 224
        loss_out = criterion(pred_age.squeeze(dim=-1),age)  
        loss_out.backward()
        optimizer.step()
        
        
        res.append(loss_out)
        
    tq.close()
    end = time.time()
    print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))
    
    loss_sum = 0
    for ele in res:
        loss_sum += ele

    loss_final = loss_sum / len(res)
    # print(loss_final)
    
    return loss_final
    


def val_model(model_age, criterion):
    model_age.eval()
    start = time.time()
    res = []
    tq = tqdm(val_dataloader)
    
    with torch.no_grad():
        for idx,(imgt, age,sex) in enumerate(tq):
            imgt = imgt.to(device).to(torch.float32) 
            age = age.to(device).to(torch.float32)  
            sex = sex.to(device).to(torch.float32) 
            pred_age = model_age(imgt,sex)
            # pred_age = pred_age * 224
            loss_out = criterion(pred_age.squeeze(dim=-1),age)  
            res.append(loss_out)
        
    tq.close()
    end = time.time()
    print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))
    
    loss_sum = 0
    for ele in res:
        loss_sum += ele

    loss_final = loss_sum / len(res)
    # print(loss_final)
    
    return loss_final



if __name__ == "__main__":
    
    best_val_loss = 100000
    for epoch in range(1000):
        train_loss = train_model(model_age, criterion, optimizer_conv,exp_lr_scheduler)
        val_loss = val_model(model_age, criterion)
        if val_loss < best_val_loss:
            torch.save(model_age.module.state_dict(), os.path.join('./models/model_age', '%d.pth' % epoch))
            best_val_loss = val_loss
    
        print('[%3d], train_loss:[%.5f], val_loss:[%.5f]' % (epoch, train_loss.item(), val_loss.item()))
        # print('[%3d], train_loss:[%.5f]' % (epoch, train_loss.item()))
    




