# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:49:21 2020

@author: Admin
"""
#---------------------------------------
#import library for project

import os 
import numpy

import pickle

import pandas as pd
import tensorflow as tf 

from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from keras.layers import Input, Conv1D, MaxPool1D, GlobalMaxPool1D, Dropout, Dense

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

#---------------------------------------
#load data (same path save file data for train and test)

os.chdir(r"E:\University\Senior Project\code_github\Senior_project\complete code") 
print("\n") 
print("Directory changed (open data)") 
print("\n") 

scores = {} # scores is an empty dict already

if os.path.getsize('data_lungsound_nozero.pckl') > 0:      
    with open('data_lungsound_nozero.pckl', "rb") as f:
        unpickler = pickle.Unpickler(f)
        # if file is not empty scores will be equal
        # to the value unpickled
        scores = unpickler.load()

Y = scores[0]      
X = scores[1]     
# Y_test = scores[2]     
# X_test = scores[3]     

del f
del scores
del unpickler

#---------------------------------------
#create model
def get_model(layer, active_1, active_2, active_3, Dropout_1, Dropout_2):
    
    inp = Input(shape=(16384, 1))
    
    #---------------------------------------
    #convolution part
    
    #convolution1
    img_1 = Conv1D(128, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    # img_1 = Dropout(rate=0.1)(img_1)
    
    #convolution2
    # img_1 = Conv1D(64, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    # img_1 = MaxPool1D(pool_size=2)(img_1)
    # img_1 = Dropout(rate=0.1)(img_1)
    
    # #convolution3
    img_1 = Conv1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    # img_1 = Dropout(rate=0.1)(img_1)
    
    #---------------------------------------
    #fully connected part
    #layer1
    dense_1 = Dense(500, activation=active_1, name="dense_1")(img_1)
    dense_1 = Dropout(rate=Dropout_1)(dense_1)
    
    #layer2
    dense_1 = Dense(500, activation=active_2, name="dense_2")(dense_1)
    dense_1 = Dropout(rate=Dropout_2)(dense_1)
    
    #layer2
    dense_1 = Dense(1, activation=active_3, name="output_dense")(dense_1)

    #train model
    model = models.Model(inputs=inp, outputs=dense_1)
    # opt = optimizers.Adam(0.1)

    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    model.summary()
    return model
#---------------------------------------
#run function create deep learning model 
af_1 = {0: 'relu', 1: 'softmax'}
af_2 = {0: 'relu', 1: 'softmax'}

for num_ac1 in range(0, 1, 1):
    for num_ac2 in range(0, 1, 1):
        for num_ac3 in range(1, 2, 1):
            
            print(af_1[num_ac1])
            print(af_1[num_ac2])
            print(af_2[num_ac3])
            
            for DO_1 in range(3, 4, 1):
                for DO_2 in range(0, 4, 1):
                    
                    Dropout_1 = float(format((2*DO_1+1)*0.1, '.1f'))
                    Dropout_2 = float(format((2*DO_2+1)*0.1, '.1f'))
                    layer = 3
                    active_1 = af_1[num_ac1]
                    active_2 = af_1[num_ac2]
                    active_3 = af_2[num_ac3]
                    
                    print(Dropout_1)
                    print(Dropout_2)
                    
                    #---------------------------------------
                    #test model in example code
                    model = get_model(layer, active_1, active_2, active_3, Dropout_1, Dropout_2)
                    history = model.fit(X, Y, epochs=500, verbose=1)
                    
                    #---------------------------------------
                    #create folder
                    new_path = r"E:/University/Senior Project/code_github/Senior_project/complete code/"+ "_" +str(layer)+ "_" +active_1+ "_" +active_2+ "_" +active_3+ "_" +str(Dropout_1)+ "_" +str(Dropout_2)
                    os.mkdir(new_path)
                    
                    #---------------------------------------
                    #save model
                    
                    os.chdir(new_path) 
                    print("Directory changed (save model)") 
                    name_model = "classify_lungsound"+ "_" +str(layer)+ "_" +active_1+ "_" +active_2+ "_" +active_3+ "_" +str(Dropout_1)+ "_" +str(Dropout_2)
                    file_path = name_model+ ".h5"
                    model.save(file_path)
                    
                    #---------------------------------------
                    #set path save file data for train and test
                    
                    os.chdir(new_path) 
                    print("\n") 
                    print("Directory changed (save result train)") 
                    print("\n") 
                    
                    #---------------------------------------
                    #save file train and test
                    
                    name_result = 'result_train_lungsound'+ "_" +str(layer)+ "_" +active_1+ "_" +active_2+ "_" +active_3+ "_" +str(Dropout_1)+ "_" +str(Dropout_2)
                    f = open(name_result +'.pckl', 'wb')
                    pickle.dump(history.history, f)
                    f.close()
                    
                    del model

