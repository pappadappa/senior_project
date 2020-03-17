# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:11:33 2020

@author: Admin
"""
#---------------------------------------
#import library for project

import pandas as pd
import numpy as np
import os 
import pickle

import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix

#---------------------------------------
#load data (same path save file data for train and test)

os.chdir(r"E:\University\Senior Project\code_github\Senior_project\complete code") 
print("\n") 
print("Directory changed (open data)") 
print("\n") 

scores = {} # scores is an empty dict already

if os.path.getsize('data_lungsound.pckl') > 0:      
    with open('data_lungsound.pckl', "rb") as f:
        unpickler = pickle.Unpickler(f)
        # if file is not empty scores will be equal
        # to the value unpickled
        scores = unpickler.load()
    
Y_test = scores[2]     
X_test = scores[3]     

#---------------------------------------
#open model

os.chdir(r"E:\University\Senior Project\code_github\Senior_project") 
print("Directory changed (open model)") 

model = load_model("classify_lungsound.h5")

#---------------------------------------
#test model

pred_test = model.predict(X_test)

#---------------------------------------
#post-processing

fpr, tpr, thresholds = roc_curve(Y_test, pred_test)

def Find_Optimal_Cutoff(target, predicted):
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(thresholds, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

# Find optimal probability threshold
threshold_se = Find_Optimal_Cutoff(Y_test, pred_test)
print(threshold_se)

pred_testre = (pred_test>threshold_se).astype(np.int8)

acc = accuracy_score(Y_test, pred_testre)
print(acc)
con = confusion_matrix(Y_test, pred_testre, normalize=None)
print(con)

#---------------------------------------
#set path save result test

os.chdir(r"E:\University\Senior Project\code_github\Senior_project\complete code") 
print("\n") 
print("Directory changed (save result test)") 
print("\n") 

#---------------------------------------
#save file train and test

f = open('test_lungsound.pckl', 'wb')
pickle.dump([acc, con, fpr, tpr, thresholds, threshold_se, pred_test, pred_testre ], f)
f.close()