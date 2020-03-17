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

import tensorflow as tf
from keras.layers import Input, Conv1D, MaxPool1D, Dense, Dropout, GlobalMaxPool1D
from keras import optimizers, losses, activations, models

tf.keras.backend.clear_session()

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

Y = scores[0]      
X = scores[1]     
# Y_test = scores[2]     
# X_test = scores[3]     

del f
del scores
del unpickler

#---------------------------------------
#create model
def get_model():
    
    nclass = 1
    inp = Input(shape=(262144, 1))
    
    #---------------------------------------
    #convolution part
    
    #convolution1
    img_1 = Conv1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = Conv1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    
    #convolution2
    img_1 = Conv1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Conv1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    
    # #convolution3
    img_1 = Conv1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Conv1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)
    
    #---------------------------------------
    #fully connected part
    
    #layer1
    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    
    #layer2
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    
    #layer3
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3_ptbdb")(dense_1)

    #train model
    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    model.compile(optimizer=opt, 
                  loss=losses.binary_crossentropy, 
                  metrics=['acc'])
    model.summary()
    return model

#---------------------------------------
#run function create deep learning model 
    
model = get_model()

#---------------------------------------
#train model in example code

# checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
# redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
# callbacks_list = [checkpoint, early, redonplat]  # early

# history = model.fit(X, Y, epochs=2, verbose=1, callbacks=callbacks_list, validation_split=0.1)

history = model.fit(X, Y, epochs=2, verbose=1)

#---------------------------------------
#save model

os.chdir(r"E:\University\Senior Project\code_github\Senior_project") 
print("Directory changed (save model)") 
file_path = "classify_lungsound.h5"
model.save(file_path)

#---------------------------------------
#set path save file data for train and test

os.chdir(r"E:\University\Senior Project\code_github\Senior_project\complete code") 
print("\n") 
print("Directory changed (save result)") 
print("\n") 

#---------------------------------------
#save file train and test

f = open('result_train_lungsound.pckl', 'wb')
pickle.dump(history.history, f)
f.close()

