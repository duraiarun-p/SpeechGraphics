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
from tensorflow.keras import backend as K
from tensorflow.keras.models import clone_model
from tensorflow.keras.utils import Sequence

# cfg = tf.compat.v1.ConfigProto() 
# cfg.gpu_options.allow_growth = True
# sess= tf.compat.v1.Session(config=cfg)

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
    idx=np.random.randint(len(label_lis_total)-batch_size)
    batch_wave_lis=wave_lis_total[idx:idx+batch_size]
    batch_emo_label_index_lis=emo_label_index_arr_total[idx:idx+batch_size]
    # print(idx)
    return batch_wave_lis, batch_emo_label_index_lis
# Convert to tensors
def tensorize_vectors(data,data_label_index):
    data_tf=tf.convert_to_tensor(data)
    data_tf=tf.expand_dims(data_tf, axis=0)
    data_tf=tf.expand_dims(data_tf, axis=-1)
    padlength=data_tf.shape
    HalfLength=(AudioLength-padlength[1])//2
    data_tf=tf.keras.layers.ZeroPadding1D(padding=HalfLength)(data_tf)
    padlength1=data_tf.shape
    NewPadlength=(AudioLength-padlength1[1])
    data_tf=tf.keras.layers.ZeroPadding1D(padding=(0,NewPadlength))(data_tf)
    data_label=np.zeros(No_of_classes)
    data_label[data_label_index]=1
    data_label_tf=tf.convert_to_tensor(data_label)
    return data_tf,data_label_tf

# for att in range(10):
#     batch_wave_lis, batch_emo_label_index_lis=batch_load(wave_lis_total,emo_label_index_arr_total,10)
#     print(att)
#%% Dataset Partition

#%%
batch_size=1
batch_wave_lis, batch_emo_label_index_lis=batch_load(wave_lis_total,emo_label_index_arr_total,batch_size)
samplerate, data = wavfile.read(batch_wave_lis[0])# For every item in the batch
data_label_index = batch_emo_label_index_lis[0]
data_tf,data_label_tf=tensorize_vectors(data,data_label_index)

# d0 = keras.Input(shape=None,name='Input')
# dP = layers.Conv1D(1, kernel_size=3,strides=2, padding='same',activation='relu',input_shape=None)(d0)
# model = tf.keras.Model(inputs=d0, outputs=dP)


    
#%%
tf.keras.backend.clear_session()