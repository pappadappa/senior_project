# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 23:51:25 2020

@author: Admin
"""

import os 
import numpy as np
import pickle
import pandas as pd

path =  r"E:\University\Senior Project\code_github\Senior_project\complete code\result"

os.chdir(path) 
print("\n") 
print("Directory changed (set path)") 
print("\n") 


floderlist = os.listdir(path)
acc_all = pd.DataFrame()
loss_all = pd.DataFrame()


for num in range(0, len(floderlist), 1):
    
    newpath = os.path.join(path,floderlist[num])
    os.chdir(newpath) 
    print("\n") 
    print("Directory changed (set path for result)") 
    print("\n") 
    
    namemodel = 'result_train_lungsound' +floderlist[num]+ '.pckl'
    
    if os.path.getsize(namemodel) > 0:      
        with open(namemodel, "rb") as f:
            unpickler = pickle.Unpickler(f)
            # if file is not empty scores will be equal
            # to the value unpickled
            scores = unpickler.load()
            
    accuracy = scores['accuracy']
    loss = scores['loss']
    
    acc = accuracy[499]
    acc = [acc]
    p_acc = pd.DataFrame(acc)
    acc_all = pd.concat([acc_all, p_acc], ignore_index=True)   
    
    loss = loss[499]
    loss = [loss]
    p_loss = pd.DataFrame(acc)
    loss_all = pd.concat([loss_all, p_loss], ignore_index=True) 
    