# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:27:01 2020

@author: Admin
"""


import pandas as pd
import numpy as np
import os 
import pickle

import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import accuracy_score , confusion_matrix, roc_curve

#---------------------------------------
#load data1 (same path save file data for train and test)

path =  r"E:\University\Senior Project\code_github\Senior_project\complete code\result"

os.chdir(path) 
print("\n") 
print("Directory changed (set path)") 
print("\n") 


floderlist = os.listdir(path)
acc_all = pd.DataFrame()
con_all = pd.DataFrame()
    
    
# for num in range(60, len(floderlist), 1):
for num in range(0, 60, 1):

    newpath = os.path.join(path,floderlist[num])
    os.chdir(newpath) 
    print("\n") 
    print("Directory changed (set path for result)") 
    print("\n") 
    
    name1 = 'acc' +floderlist[num]+ '.pckl'
    
    if os.path.getsize(name1) > 0:      
        with open(name1, "rb") as f:
            unpickler = pickle.Unpickler(f)
            # if file is not empty scores will be equal
            # to the value unpickled
            scores1 = unpickler.load()
    
    name2 = 'con' +floderlist[num]+ '.pckl'
        
    if os.path.getsize(name2) > 0:      
        with open(name2, "rb") as f:
            unpickler = pickle.Unpickler(f)
            # if file is not empty scores will be equal
            # to the value unpickled
            scores2 = unpickler.load()
            
    acc = [scores1]
    con = scores2
    
    p_acc = pd.DataFrame(acc)
    acc_all = pd.concat([acc_all, p_acc], ignore_index=True) 
    
    p_con1 = pd.DataFrame(con[0])
    p_con2 = pd.DataFrame(con[1])
    p_con = pd.concat([p_con1, p_con2], ignore_index=True)
    p_con = p_con.T
    con_all = pd.concat([con_all, p_con], ignore_index=True) 
    
    