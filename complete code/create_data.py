# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:12:34 2020
test code beautiful
@author: Admin
"""
#---------------------------------------
#import library for project

import pandas as pd
import numpy as np
import os 

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import pickle


#---------------------------------------
#change path to crackle

os.chdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\Without Zero\FFT_Crackle\power") 
print("\n")
print("Directory changed (Crackle)") 
print("\n") 

#---------------------------------------
#create Dataframe of crackle

entries = os.listdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\Without Zero\FFT_Crackle\power")
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

del entries
del data1
del data_raw1
del dataT1
del pdata1
del i

#---------------------------------------
#change path to wheeze
    
os.chdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\Without Zero\FFT_Wheeze\power") 
print("\n") 
print("Directory changed (Wheeze)") 
print("\n") 

#---------------------------------------
#create Dataframe of wheeze

entries = os.listdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\Without Zero\FFT_Wheeze\power")
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

del entries
del data2
del data_raw2
del dataT2
del pdata2
del i

#---------------------------------------
#separate file train and test

dfp_train1, dfp_test1, dft_train1, dft_test1 = train_test_split(df_p1, df_t1, test_size=0.1, random_state=None)
dfp_train2, dfp_test2, dft_train2, dft_test2 = train_test_split(df_p2, df_t2, test_size=0.1, random_state=None)

del df_p1
del df_p2
del df_t1
del df_t2

dfp_train = pd.concat([dfp_train1,dfp_train2])
dft_train = pd.concat([dft_train1,dft_train2]) 
dfp_test = pd.concat([dfp_test1,dfp_test2]) 
dft_test = pd.concat([dft_test1,dft_test2]) 

del dfp_train1
del dfp_train2
del dfp_test1
del dfp_test2
del dft_train1
del dft_train2
del dft_test1
del dft_test2

dfp_train['result'] = pd.Series(dft_train[0], index=dfp_train.index)
df_train = dfp_train
dfp_test['result'] = pd.Series(dft_test[0], index=dfp_test.index)
df_test = dfp_test

del dfp_train
del dft_train
del dfp_test
del dft_test

df_train = df_train.sample(frac=1, replace=False, random_state=1)
df_test = df_test.sample(frac=1, replace=False, random_state=1)

#---------------------------------------
#tranforms file with zero
# Y = np.array(df_train['result'].values)
# Y = np.reshape(Y, (Y.shape[0],))
# X = np.array(df_train[list(range(262144))].values).astype('float32')
# X = np.reshape(X, (X.shape[0], 262144, 1))

# Y_test = np.array(df_test['result'].values)
# Y_test = np.reshape(Y_test, (Y_test.shape[0],))
# X_test = np.array(df_test[list(range(262144))].values).astype('float32')
# X_test = np.reshape(X_test, (X_test.shape[0], 262144, 1))

#---------------------------------------
#tranforms file without zero
Y = np.array(df_train['result'].values)
Y = np.reshape(Y, (Y.shape[0],))
X = np.array(df_train[list(range(16384))].values).astype('float32')
X = np.reshape(X, (X.shape[0], 16384, 1))

Y_test = np.array(df_test['result'].values)
Y_test = np.reshape(Y_test, (Y_test.shape[0],))
X_test = np.array(df_test[list(range(16384))].values).astype('float32')
X_test = np.reshape(X_test, (X_test.shape[0], 16384, 1))

del df_train
del df_test

#---------------------------------------
#set path save file data for train and test

os.chdir(r"E:\University\Senior Project\code_github\Senior_project\complete code") 
print("\n") 
print("Directory changed (save data)") 
print("\n") 

#---------------------------------------
#save file train and test

f = open('data_lungsound_nozero.pckl', 'wb')
pickle.dump([Y, X, Y_test, X_test], f)
f.close()