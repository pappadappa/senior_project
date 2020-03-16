# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:54:51 2020

@author: Chonthicha Pinkaraket
"""
#---------------------------------------
#import library for project

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

os.chdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\Save_S_output_Crackle\power") 
print("Directory changed (Crackle)") 
print("\n") 

#---------------------------------------
#create Dataframe of crackle

entries = os.listdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\Save_S_output_Crackle\power")
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
    
os.chdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\Save_S_output_Wheeze\power") 
print("Directory changed (Wheeze)") 
print("\n") 

#---------------------------------------
#create Dataframe of wheeze

entries = os.listdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\Save_S_output_Wheeze\power")
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

df_p = pd.concat([df_p1, df_p2])
df_t = pd.concat([df_t1, df_t2])

del df_p1
del df_p2
del df_t1
del df_t2

#---------------------------------------
#separate file train and test

dfp_train, dfp_test, dft_train, dft_test = train_test_split(df_p, df_t, test_size=0.1, random_state=None)

del df_p
del df_t

Y = np.array(dft_train.values)
X = np.array(dfp_train.values).astype('float32')
X = np.reshape(X, (X.shape[0], 1, 262144))

Y_test = np.array(dft_test.values)
X_test = np.array(dfp_test.values).astype('float32')
X_test = np.reshape(X_test, (X_test.shape[0], 1, 262144))

del dfp_train
del dfp_test
del dft_train
del dft_test

# p = df_p.values.tolist()
# t = df_t.values.tolist()

# kf = KFold(n_splits=5, shuffle=True)
# kf.get_n_splits(df_p)

# print(kf)

# X = []
# Y = []
# X_test = []
# Y_test = []

# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     print("\n")
#     X, X_test = X[train_index], X[test_index]
#     Y, Y_test = y[train_index], y[test_index]

#---------------------------------------
#function create deep learning model 

def get_model():
    
    nclass = 1
    inp = Input(shape=(1,524288))
    
    #---------------------------------------
    #convolution part
    
    #convolution1
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    
    #convolution2
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    
    #convolution3
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    
    #convolution4
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    
    #---------------------------------------
    #fully connected part
    
    #layer1
    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    
    #layer2
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    
    #layer3
    dense_1 = Dense(nclass, activation=activations.sigmoid, name="dense_3_ptbdb")(dense_1)

    #train model
    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    model.summary()
    return model

#---------------------------------------
#run function create deep learning model 
    
model = get_model()

#---------------------------------------
#save model

os.chdir(r"E:\University\Senior Project\code_github\Senior_project") 
print("Directory changed (save model)") 
file_path = "classify_lungsound.h5"
model.save(file_path)

#---------------------------------------
#test model

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early
model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)
model.load_weights(file_path)

pred_test = model.predict(X_test)
pred_test = (pred_test>0.5).astype(np.int8)

f1 = f1_score(Y_test, pred_test)

print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)


