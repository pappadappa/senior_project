# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 16:43:31 2020

@author: Admin
"""
# import tables

# file = tables.openFile('f_FFT_Nor_106_2b1_Pl_mc_LittC2SE_1.mat')
# lon = file.root.lon[:]
# lat = file.root.lat[:]
# # Alternate syntax if the variable name is in a string
# varname = 'lon'
# lon = file.getNode('/' + varname)[:]

# from scipy.io import loadmat
# x = loadmat('1.mat')
# lon = x['lon']
# lat = x['lat']
# # one-liner to read a single variable
# lon = loadmat('test.mat')['lon']

# from os.path import dirname, join as pjoin
# import scipy.io as sio


# path = 

# data_dir = pjoin(dirname(path), 'matlab', 'tests', 'data')
# mat_fname = pjoin(data_dir, '1.mat')
# mat_contents = sio.loadmat(mat_fname)


# print(mat_contents['testdouble'])


#---------------------------------------
#open .mat succes

import pandas as pd
import numpy as np
import tensorflow as tf
import os 

from scipy.io import loadmat
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
  
# change the current directory 
os.chdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\Save_S_output_Crackle\power") 
print("Directory changed") 
print("\n") 

# data1 = loadmat('Nor_106_2b1_Pl_mc_LittC2SE_1.mat')
# print(data1['s1_output'])

# data_raw = data1['s1_output']
# pdata = pd.DataFrame(data_raw.T)
# print(pdata)

#---------------------------------------
#input data form fourier transform

# entries = os.listdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\FFT_Nor_Crackle")

# for i in entries:
#     data1 = loadmat(i)
#     data_raw = data1['f']
#     pdata = pd.DataFrame(data_raw)
#     all_pdata = pdata.append(pdata)

#--------------------------------------

entries = os.listdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\Save_S_output_Crackle\power")
# print(entries)
all_pdata = pd.DataFrame()

for i in range(0, len(entries), 1):
    data1 = loadmat(entries[i])
    data_raw = data1['P']
    # print(data_raw)
    dataT1 = data_raw.T
    print(i)
    pdata = pd.DataFrame(pd.DataFrame(dataT1))
    all_pdata = pd.concat([all_pdata,pdata], ignore_index=True)   

all_ndata = all_pdata.to_numpy()
all_ndata = np.reshape(all_ndata, (80, 1, 262144))

#--------------------------------------

# os.chdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\Save_S_output_Crackle\power") 
# print("Directory changed (Crackle)") 
# print("\n") 

# #---------------------------------------
# #create Dataframe of crackle

# entries = os.listdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\Save_S_output_Crackle\power")
# # print(entries)

# df_p1 = pd.DataFrame()

# for i in range(0, len(entries), 1):
#     data1 = loadmat(entries[i])
#     data_raw1 = data1['P']
#     # print(data_raw)
#     dataT1 = data_raw1.T
#     print(i)
#     pdata1 = pd.DataFrame(dataT1)
#     df_p1 = pd.concat([df_p1,pdata1], ignore_index=True) 

# df_t1 = pd.DataFrame(np.zeros(len(df_p1)))
    
# #---------------------------------------
# #change path to wheeze
    
# os.chdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\Save_S_output_Wheeze\power") 
# print("Directory changed (Wheeze)") 
# print("\n") 

# #---------------------------------------
# #create Dataframe of wheeze

# entries = os.listdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\Save_S_output_Wheeze\power")
# # print(entries)

# df_p2 = pd.DataFrame()

# for i in range(0,  len(entries), 1):
#     data2 = loadmat(entries[i])
#     data_raw2 = data2['P']
#     # print(data_raw)
#     dataT2 = data_raw2.T
#     print(i)
#     pdata2 = pd.DataFrame(dataT2)
#     df_p2 = pd.concat([df_p2,pdata2], ignore_index=True) 
    
# df_t2 = pd.DataFrame(np.ones(len(df_p2)))

# df_p = pd.concat([df_p1, df_p2])
# df_t = pd.concat([df_t1, df_t2])

# arr_p = df_p.to_numpy()
# arr_t = df_t.to_numpy()

