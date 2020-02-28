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
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

#---------------------------------------
#change path to crackle

os.chdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\FFT_Nor_Crackle") 
print("Directory changed") 
print("\n") 

#---------------------------------------
#create Dataframe of crackle

entries = os.listdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\FFT_Nor_Crackle")
# print(entries)
for i in entries:
    data1 = loadmat(i)
    data_raw = data1['f']
    pdata = pd.DataFrame(data_raw)
    df_1 = pdata.append(pdata)
    
#---------------------------------------
#change path to wheeze
    
os.chdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\FFT_Nor_Wheeze") 
print("Directory changed") 
print("\n") 

#---------------------------------------
#create Dataframe of wheeze

entries = os.listdir(r"E:\University\Senior Project\code_github\Senior_project\database form matlab\FFT_Nor_Wheeze")
# print(entries)
for i in entries:
    data1 = loadmat(i)
    data_raw = data1['f']
    pdata = pd.DataFrame(data_raw)
    df_2 = pdata.append(pdata)

df = pd.concat([df_1, df_2])

#---------------------------------------
#separate file train and test

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])

Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]


#---------------------------------------
#function create deep learning model 

def get_model():
    
    nclass = 1
    inp = Input(shape=(187, 1))
    
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


