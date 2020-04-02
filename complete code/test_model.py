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
    
    
for num in range(111, len(floderlist), 1):

    path =  r"E:\University\Senior Project\code_github\Senior_project\complete code"

    os.chdir(path) 
    print("\n") 
    print("Directory changed (set path)") 
    print("\n") 
    
    scores = {} # scores is an empty dict already
    
    namemodel = 'data_lungsound_nozero.pckl'
    if os.path.getsize(namemodel) > 0:      
        with open(namemodel, "rb") as f:
            unpickler = pickle.Unpickler(f)
            # if file is not empty scores will be equal
            # to the value unpickled
            scores = unpickler.load()
        
    Y_test = scores[2]     
    X_test = scores[3]   
    
    del scores
    
    path =  r"E:\University\Senior Project\code_github\Senior_project\complete code\result"

    os.chdir(path) 
    print("\n") 
    print("Directory changed (set path)") 
    print("\n") 
    
    #---------------------------------------
    #load data2 (same path save file data for train and test)
    
    newpath = os.path.join(path,floderlist[num])
    os.chdir(newpath) 
    print("\n") 
    print("Directory changed (open data)") 
    print("\n")
    
    #---------------------------------------
    #open model
    
    namemodel = "classify_lungsound" +floderlist[num]+ ".h5"
    model = load_model(namemodel)
    
    #---------------------------------------
    #test model
    
    pred_test = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(Y_test, pred_test)
    print(thresholds)
    
    #---------------------------------------
    #post-processing
    
    def Find_Optimal_Cutoff(target, predicted):
        i = np.arange(len(tpr)) 
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(thresholds, index=i)})
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    
        return list(roc_t['threshold']) 
    
    # Find optimal probability threshold
    threshold_se = Find_Optimal_Cutoff(Y_test, pred_test)
    print(threshold_se)
    
    del fpr
    del tpr
    del thresholds
    
    pred_testre = (pred_test>threshold_se).astype(np.int8)
    
    acc = accuracy_score(Y_test, pred_testre)
    print(acc)
    con = confusion_matrix(Y_test, pred_testre, normalize=None)
    print(con)
    
    acc = [acc]
    p_acc = pd.DataFrame(acc)
    acc_all = pd.concat([acc_all, p_acc], ignore_index=True) 
    
    p_con1 = pd.DataFrame(con[0])
    p_con2 = pd.DataFrame(con[1])
    p_con = pd.concat([p_con1, p_con2], ignore_index=True)
    p_con = p_con.T
    con_all = pd.concat([con_all, p_con], ignore_index=True)   
    
    del p_con1
    del p_con2
    del p_con
    
    #---------------------------------------
    #save file train and test
    
    namemodel = 'pred_test' +floderlist[num]+ '.pckl'
    f = open(namemodel, 'wb')
    pickle.dump(pred_test, f)
    f.close()
    
    namemodel = 'threshold_se' +floderlist[num]+ '.pckl'
    f = open(namemodel, 'wb')
    pickle.dump(threshold_se, f)
    f.close()
    
    namemodel = 'acc' +floderlist[num]+ '.pckl'
    f = open(namemodel, 'wb')
    pickle.dump(acc, f)
    f.close()
    
    namemodel = 'con' +floderlist[num]+ '.pckl'
    f = open(namemodel, 'wb')
    pickle.dump(con, f)
    f.close()
    
    del pred_test
    del threshold_se
    del acc
    del con
    
    del model