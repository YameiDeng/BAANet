# -*- coding: utf-8 -*-
"""
递归遍历所有文件
复制指定后缀文件
"""
 
import os
import csv
import shutil
import numpy as np
import pandas as pd
 
paths = ['./train','./val','./test'] # 原文件路径
postfix = ['.jpg', '.JPG', '.PNG', '.png', '.jpeg', '.JPEG'] # 指定文件后缀名

image_name1 = np.array(pd.read_csv('mark1.csv'))



for p in paths:
    pd2=[]
    items = os.listdir(p)
    
    for i,img in enumerate(image_name1):
        img_n = str(img[0])+'.jpg'
        if img_n in items:
            pd2.append(image_name1[i])
    

    pd2 = np.array(pd2)
    id1 = pd2[:,0]
    boneage = pd2[:,1]
    male = pd2[:,2]
    file = pd2[:,3]
    
    
    dataframe = pd.DataFrame({'id':id1,'boneage':boneage,'male':male,'file':file})
    dataframe.to_csv(p+".csv",index=False,sep=',')


