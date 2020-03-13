# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:41:11 2020

@author: Admin
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import os 

from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv1D

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

from sklearn.metrics import accuracy_score, f1_score

#---------------------------------------
#change path to crackle

os.chdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\FFT_Nor_Crackle\power") 
print("Directory changed (Crackle)") 
print("\n") 

#---------------------------------------
#create Dataframe of crackle

entries = os.listdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\FFT_Nor_Crackle\power")
# print(entries)

df_p1 = pd.DataFrame()

for i in range(0, len(entries), 1):
    data1 = loadmat(entries[i])
    data_raw1 = data1['P']
    # print(data_raw)
    dataT1 = data_raw1.T
    print(i)
    pdata1 = pd.DataFrame(dataT1)
    df_p1 = pd.concat([df_p1,pdata1], ignore_index=True) 

df_t1 = pd.DataFrame(np.zeros(len(df_p1)))
    
#---------------------------------------
#change path to wheeze
    
os.chdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\FFT_Nor_Wheeze\power") 
print("Directory changed (Wheeze)") 
print("\n") 

#---------------------------------------
#create Dataframe of wheeze

entries = os.listdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\FFT_Nor_Wheeze\power")
# print(entries)

df_p2 = pd.DataFrame()

for i in range(0,  len(entries), 1):
    data2 = loadmat(entries[i])
    data_raw2 = data2['P']
    # print(data_raw)
    dataT2 = data_raw2.T
    print(i)
    pdata2 = pd.DataFrame(dataT2)
    df_p2 = pd.concat([df_p2,pdata2], ignore_index=True) 
    
df_t2 = pd.DataFrame(np.ones(len(df_p2)))

df_p = pd.concat([df_p1, df_p2])
df_t = pd.concat([df_t1, df_t2])


#---------------------------------------
#separate file train and test

dfp_train, dfp_test, dft_train, dft_test = train_test_split(df_p, df_t, test_size=0.1, random_state=None)

Y = np.array(dft_train.values)
X = np.array(dfp_train.values).astype('float32')

Y_test = np.array(dft_test.values)
X_test = np.array(dfp_test.values).astype('float32')

#---------------------------------------

# model = Sequential()
# model.add(Conv1D(51200, 64, 
#           activation='relu', 
#           input_shape=(1,1048576)))
