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
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

# from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import BoneDataset
from resnet50 import AgePredictor
import pandas as pd

torch.manual_seed(2022)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_color = [(0),(255)]          

results_path = './results'
exp_name = 'TASNet'

parser = argparse.ArgumentParser(description='feature extarct')
parser.add_argument('--batch_size', type=int, default=2, 
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



train_dataset_path = './data_e/train'
val_dataset_path = './data_e/val'
test_dataset_path = './data_e/test'

train_csv_path = './data_e/train.csv'
val_csv_path = './data_e/val.csv'
test_csv_path = './data_e/test.csv'


    # data
train_dataset = BoneDataset(data_path=train_dataset_path, csv_path=train_csv_path, train=True)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

val_dataset = BoneDataset(data_path=val_dataset_path, csv_path=val_csv_path, train=False)
val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=0)

test_dataset = BoneDataset(data_path=test_dataset_path, csv_path=test_csv_path, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0)
   
print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset),"test_datasize", len(test_dataset))
backbone_path = './backbone/resnet/resnet50-19c8e357.pth'




    


def val_model(model_age, criterion):
    model_age.eval()
    start = time.time()
    res = []
    tq = tqdm(test_dataloader)
    
    result = []
    with torch.no_grad():
        for idx,(imgt, age,sex) in enumerate(tq):
            imgt = imgt.to(device).to(torch.float32) 
            age = age.to(device).to(torch.float32)  
            sex = sex.to(device).to(torch.float32) 
            pred_age = model_age(imgt,sex)
            result.append([age.squeeze(dim=-1).cpu().numpy(),pred_age.squeeze(dim=-1).squeeze(dim=-1).cpu().numpy()])
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
    
    name=['Age','Pre']
    result=pd.DataFrame(columns=name,data=result)#数据有三列，列名分别为one,two,three
    result.to_csv('resnet_e.csv',encoding='gbk') 
    
    return loss_final



if __name__ == "__main__":
    model_age = AgePredictor(num_classes=1).to(device)
    #model_age = nn.DataParallel(model_age)
    criterion =  torch.nn.L1Loss() 
    
    
    epoch = 185
    path = os.path.join('./models/model_age', '%d.pth' % epoch)
    model_age.load_state_dict(torch.load(path))
    # checkpoint = torch.load(path,map_location=device)
    # model_age.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()}, strict=False)
    
    
    val_loss = val_model(model_age, criterion)
    print('[%3d], test_loss:[%.5f]' % (epoch, val_loss.item()))
    

    




