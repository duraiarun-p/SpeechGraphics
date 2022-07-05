#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 20:47:36 2022

@author: arun
"""

import re
import os
import pickle

#%%

file_path='/home/arun/Documents/PyWSPrecision/datasets/IEMOCAP_full_release/Session1/dialog/EmoEvaluation/Ses01F_impro01.txt'
data_path='/home/arun/Documents/PyWSPrecision/datasets/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01'


def wave_with_label_per_record_per_session(file_path,data_path):  
    useful_regex = re.compile(r'\[.+\]\n', re.IGNORECASE)
    
    with open(file_path) as f:
        file_content = f.read()
        
    info_lines = re.findall(useful_regex, file_content)
    
    del info_lines[0]
    
    label_lis=[]
    labels_arr=[]
    lines_len=len(info_lines)
    
    for l in info_lines:
        labels=(l.strip().split('\t'))
        labels_arr=[labels[1],labels[2]]
        # labels_arr[0]=labels[1]
        # labels_arr[1]=labels[2]
        label_lis.append(labels_arr)
    
    wave_lis=[]# List of wave files per session
    # file_lis=os.listdir(data_path)
    for file in os.listdir(data_path):
        if not file.startswith(".") and file.endswith(".wav"):
            wave_lis.append(data_path+'/'+file)
    wave_lis.sort()
    
    emotion_arr=[]# List of labels per session
    lines_len=len(label_lis)
    for fi in range(lines_len):
        #file_lis[fi,1]=
        emotion_arr.append(label_lis[fi][1])
    
    return wave_lis,emotion_arr,label_lis

#%%

# wave_lis,emo_label_arr=wave_with_label_per_record_per_session(emo_eval_file_path,wav_folder_path)

session_path='/home/arun/Documents/PyWSPrecision/datasets/IEMOCAP_full_release/'

session_lis_total=[]
for file in os.listdir(session_path):
    if file.startswith("Session"):
        session_lis_total.append(file)
session_lis_total.sort()
sess_total_len=len(session_lis_total)

wave_lis_total=[]
emo_label_arr_total=[]
label_lis_total=[]

for sessi in range(sess_total_len):
    emo_eval_lis=[]
    emo_eval_lis_fname=[]
    emo_eval_path=session_path+session_lis_total[sessi]+'/dialog/EmoEvaluation/' # Note session index loop iteration
    for file in os.listdir(emo_eval_path):
        if file.startswith("Ses"):
            fname=file.split('.txt')
            emo_eval_lis.append(file)
            emo_eval_lis_fname.append(fname[0])
    emo_eval_lis.sort()
    emo_eval_lis_fname.sort()
    emo_len_per_sess=len(emo_eval_lis)
    
    for emoi in range(emo_len_per_sess):    
        emo_eval_file_path=session_path+session_lis_total[sessi]+'/dialog/EmoEvaluation/'+emo_eval_lis[emoi]
        wav_folder_path=session_path+session_lis_total[sessi]+'/sentences/wav/'+emo_eval_lis_fname[emoi] # Note emo eval index loop iteration
        wave_lis,emo_label_arr,label_lis=wave_with_label_per_record_per_session(emo_eval_file_path,wav_folder_path)
        wave_lis_total=wave_lis_total+wave_lis
        emo_label_arr_total=emo_label_arr_total+emo_label_arr
        label_lis_total=label_lis_total+label_lis
        # print(emoi)    
    # print(sessi)

emo_cat_lis=list(set(emo_label_arr_total))
emo_cat_lis.sort()

emo_label_index_arr_total=[]
emo_label_arr_total_len=len(emo_label_arr_total)

for emoi in range(emo_label_arr_total_len):
    label_idx=emo_cat_lis.index(emo_label_arr_total[emoi])
    emo_label_index_arr_total.append(label_idx)

#%% Saving file paths and corresponding labels as alternate database
file_name = "WaveFilePathsTotal.pkl"
open_file = open(file_name, "wb")
pickle.dump(wave_lis_total, open_file)
open_file.close()

file_name = "EmoFileNameLabelTotal.pkl"
open_file = open(file_name, "wb")
pickle.dump(label_lis_total, open_file)
open_file.close()

file_name = "EmoLabelTextArrayTotal.pkl"
open_file = open(file_name, "wb")
pickle.dump(emo_label_arr_total, open_file)
open_file.close()

file_name = "EmoLabelNumericArrayTotal.pkl"
open_file = open(file_name, "wb")
pickle.dump(emo_label_index_arr_total, open_file)
open_file.close()

file_name = "EmoLabelCategoryTotal.pkl"
open_file = open(file_name, "wb")
pickle.dump(emo_cat_lis, open_file)
open_file.close()