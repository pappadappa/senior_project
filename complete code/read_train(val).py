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

vacc_all = pd.DataFrame()
vloss_all = pd.DataFrame()

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
    val_accuracy = scores['val_accuracy']
    val_loss = scores['val_loss']
    
    acc = accuracy[499]
    acc = [acc]
    p_acc = pd.DataFrame(acc)
    acc_all = pd.concat([acc_all, p_acc], ignore_index=True)   
    
    loss = loss[499]
    loss = [loss]
    p_loss = pd.DataFrame(loss)
    loss_all = pd.concat([loss_all, p_loss], ignore_index=True) 
    
    val_acc = val_accuracy[499]
    val_acc = [val_acc]
    p_vacc = pd.DataFrame(val_acc)
    vacc_all = pd.concat([vacc_all, p_vacc], ignore_index=True)   
    
    val_loss = val_loss[499]
    val_loss = [val_loss]
    p_vloss = pd.DataFrame(val_loss)
    vloss_all = pd.concat([vloss_all, p_vloss], ignore_index=True) 