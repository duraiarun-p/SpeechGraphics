#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 12:55:44 2022

@author: arun
"""
import os
import pickle
import numpy as np
from scipy.io import wavfile

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras.models import clone_model
from tensorflow.keras.utils import Sequence
import datetime
import time

# cfg = tf.compat.v1.ConfigProto() 
# cfg.gpu_options.allow_growth = False
# sess= tf.compat.v1.Session(config=cfg)
# with tf.Session() as sess:
# tf.device("/cpu:0")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') 
start_time_0=time.time()
#%% Load Alternate Database
file_name = "WaveFilePathsTotal.pkl"
open_file = open(file_name, "rb")
wave_lis_total=pickle.load(open_file)
open_file.close()

file_name = "EmoFileNameLabelTotal.pkl"
open_file = open(file_name, "rb")
label_lis_total=pickle.load(open_file)
open_file.close()

file_name = "EmoLabelTextArrayTotal.pkl"
open_file = open(file_name, "rb")
emo_label_arr_total=pickle.load(open_file)
open_file.close()

file_name = "EmoLabelNumericArrayTotal.pkl"
open_file = open(file_name, "rb")
emo_label_index_arr_total=pickle.load(open_file)
open_file.close()

file_name = "EmoLabelCategoryTotal.pkl"
open_file = open(file_name, "rb")
emo_cat_lis=pickle.load(open_file)
open_file.close()
#%% Input Lengths
wavlen=len(emo_label_arr_total)
MaxinputLen=[]
for emoi in range(wavlen):
    # print(emoi)
    samplerate1, data_temp = wavfile.read(wave_lis_total[emoi])
    MaxinputLen.append(len(data_temp))
AudioLength=max(MaxinputLen) #Input shape
No_of_classes=len(emo_cat_lis) # No. of classes
#%% My Data Loader
def batch_load(wave_lis_total,emo_label_index_arr_total,batch_size):
# batch_size=10
    idx=np.random.randint(len(wave_lis_total)-batch_size)
    batch_wave_lis=wave_lis_total[idx:idx+batch_size]
    batch_emo_label_index_lis=emo_label_index_arr_total[idx:idx+batch_size]
    # print(idx)
    return batch_wave_lis, batch_emo_label_index_lis
# Convert to tensors
def tensorize_vectors(data,data_label_index):
    data_n=(data-(sum(data)/len(data)))/max(data)
    data_tf=tf.convert_to_tensor(data_n)
    data_tf=tf.expand_dims(data_tf, axis=0)
    data_tf=tf.expand_dims(data_tf, axis=-1)
    # data_tf=tf.expand_dims(data_tf, axis=0)
    padlength=data_tf.shape
    HalfLength=(AudioLength-padlength[1])//2
    data_tf=tf.keras.layers.ZeroPadding1D(padding=HalfLength)(data_tf)
    padlength1=data_tf.shape
    NewPadlength=(AudioLength-padlength1[1])
    data_tf=tf.keras.layers.ZeroPadding1D(padding=(0,NewPadlength))(data_tf)
    data_label=np.zeros(No_of_classes)
    data_label[data_label_index]=1
    data_label_tf=tf.convert_to_tensor(data_label)
    data_label_tf=tf.expand_dims(data_label_tf, axis=0)
    # data_tf=tf.stack(data_tf)
    # data_label_tf=tf.stack(data_label_tf)
    return data_tf,data_label_tf,data_label

class SpeechClassifier():
    def __init__(self,AudioLength,firstfilter,No_of_classes):
        self.AudioLength=AudioLength
        self.firstfilter=firstfilter
        self.No_of_classes=No_of_classes
        self.Classifier=self.build_classifier()
        self.optimiser = keras.optimizers.Adam(0.01, 0.5,0.999)
        self.Classifier.compile(loss='mse', optimizer=self.optimiser, metrics=['accuracy'])
        
    def build_classifier(self):
        # d0 = keras.Input(shape=(1,self.AudioLength,1),name='Input')
        d0 = keras.Input(shape=(self.AudioLength,1),name='Input')
        dP = layers.Conv1D(self.firstfilter*1, kernel_size=3,strides=2, padding='same',activation='relu')(d0)
        dP = layers.MaxPooling1D(pool_size=1, strides=1, padding='valid')(dP)
        dP = activations.tanh(dP)
        dP = layers.Conv1D(self.firstfilter//2, kernel_size=3,strides=2, padding='same',activation='relu')(dP)
        dP = layers.MaxPooling1D(pool_size=1, strides=1, padding='valid')(dP)
        dP = activations.tanh(dP)
        dP = layers.Conv1D(self.firstfilter//4, kernel_size=3,strides=2, padding='same',activation='relu')(dP)
        dP = layers.MaxPooling1D(pool_size=1, strides=1, padding='valid')(dP)
        dP = activations.tanh(dP)
        dP = layers.Flatten()(dP)
        dP = layers.Dense(1024, activation='relu')(dP)
        dP = layers.Dense(512, activation='relu')(dP)
        dP = layers.Dense(128, activation='relu')(dP)
        dP = layers.Dropout(0.5)(dP)
        dP = layers.Dense(self.No_of_classes, activation='sigmoid')(dP)
        return keras.Model(d0,dP)

#%% Classifier Model
SPC=SpeechClassifier(AudioLength,50,No_of_classes)
SPC.Classifier.summary()

#%% Dataset Partition
Total_len=len(wave_lis_total)
train_stop=round(Total_len*0.7)
test_stop=Total_len-train_stop
wave_lis_total_train=wave_lis_total[0:train_stop]
emo_label_index_arr_total_train=emo_label_index_arr_total[0:train_stop]
wave_lis_total_test=wave_lis_total[-test_stop:]
emo_label_index_arr_total_test=emo_label_index_arr_total[-test_stop:]
#%% Classifier Training and Test loss

#Train step
batch_size=10
train_loss=[]
test_loss=[]
test_loss_per_batch=[]
train_loss_per_all_batch_lis=[]
train_loss_per_batch=[]
train_loss_per_batch_iter_lis=[]
wave_lis_total_train_len=len(wave_lis_total_train)//batch_size
# wave_lis_total_train_len=10 # For code testing purposes
epochs=100
# learning_rates=np.logspace(-8, 10,num=epochs) # To identify optimum learning rates
for epochi in range(epochs):
    # Training loss section
    for datai in range(wave_lis_total_train_len):
        batch_wave_lis, batch_emo_label_index_lis=batch_load(wave_lis_total_train,emo_label_index_arr_total_train,batch_size)
        for batchi in range(batch_size): 
            samplerate, data = wavfile.read(batch_wave_lis[batchi])# For every item in the batch
            data_label_index = batch_emo_label_index_lis[batchi]
            data_tf,data_label_tf,data_label=tensorize_vectors(data,data_label_index)
            train_loss_per_batch_iter=SPC.Classifier.train_on_batch(data_tf,data_label_tf)# Train
            train_loss_per_batch_iter_lis.append(train_loss_per_batch_iter[0])
        train_loss_per_batch=sum(train_loss_per_batch_iter_lis)/len(train_loss_per_batch_iter_lis)
        train_loss_per_all_batch_lis.append(train_loss_per_batch)
    train_loss_per_all_batch=sum(train_loss_per_all_batch_lis)/len(train_loss_per_all_batch_lis)
    train_loss.append(train_loss_per_all_batch)
    # Test loss section
    batch_wave_lis_test, batch_emo_label_index_lis_test=batch_load(wave_lis_total_test,
                                                                   emo_label_index_arr_total_test,batch_size//2)
    for batchi in range(batch_size//2):
        samplerate, data_test = wavfile.read(batch_wave_lis_test[batchi])# For every item in the batch
        data_test_label_index = batch_emo_label_index_lis_test[batchi]
        data_test_tf,data_test_label_index_tf,data_label=tensorize_vectors(data_test,data_test_label_index)
        test_loss_ele=SPC.Classifier.test_on_batch(data_test_tf,data_test_label_index_tf)# Test
        test_loss_per_batch.append(test_loss_ele[0])
    test_loss_per_batch_all=sum(test_loss_per_batch)/len(test_loss_per_batch)
    test_loss.append(test_loss_per_batch_all)
    print('Epoch=%s'%epochi)

#%% Save losses in mat
from scipy.io import savemat
mdic={"train_loss":train_loss,"test_loss":test_loss}
# mdic={"train_loss":train_loss,"test_loss":test_loss,"learning_rates":learning_rates}
savemat('Losses100.mat', mdic)


#%%
tf.keras.backend.clear_session()
print('Script started at')
print(st_0)
runtimeN0=(time.time()-start_time_0)/60
print('Script Total Time =%s min'%(runtimeN0))
print('Script ended at')
st_0 = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
print(st_0)