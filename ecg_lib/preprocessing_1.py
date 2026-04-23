# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:46:24 2026

@author: Matthew Miller

#fp
"""
#general first
import torch
import numpy as np

#mine second
from .ECG_Dataset_0 import ECG_Dataset

def d_average_slice_0(inputs):
    
#inputs = data[0][0]
    
    len_streams, num_streams = inputs.shape
    out_primitive=np.zeros((1, len_streams))
    
    for sample in range(len_streams):
        out_primitive[0,sample] = inputs[sample,:].sum()/num_streams
        
    return out_primitive    


def d_average_array_0(inputs, config):
    num_samples = len(inputs)
    out = np.zeros((num_samples, 1, 1000, config["data"]["num_leads"]))
    #print(f"out.shape {out.shape}")
    #print(f"len(inputs) {len(inputs)}")
    #print(f"type(inputs) {type(inputs)}")
    #print(f"len(inputs[0]) {len(inputs[0])}")
    #print(f"type(inputs[0]) {type(inputs[0])}")
    #print(f"inputs[0][0].shape {inputs[0][0].shape}")

    for sample_index in range(num_samples):
        for lead_index in range(config["data"]["num_leads"]):
            out[sample_index, 0] = inputs[sample_index][0][lead_index]

    #print(f"out.shape {out[0,0].shape}")
    return out

def preprocessing_fun_0(inputs, config):
    #unpack inputs if necessary
    X_train = inputs["X_train"]
    X_val = inputs["X_val"]
    X_test = inputs["X_test"]
    Y_train = inputs["Y_train"]
    Y_val = inputs["Y_val"]
    Y_test = inputs["Y_test"]
    # %%
    #===========================================================
    #Do data stuff
    #need to fix the data into an appropriate pytorch tensor
    #Since the data has already been divided into 3 folds, I will be brute forcing this

    X_train_0 = d_average_array_0(X_train, config)
    X_val_0 = d_average_array_0(X_val, config)
    X_test_0 = d_average_array_0(X_test, config)

    X_train_1 = torch.tensor(X_train_0, dtype=torch.float32)
    X_val_1 = torch.tensor(X_val_0, dtype=torch.float32)
    X_test_1 = torch.tensor(X_test_0, dtype=torch.float32)

    train_dataset = ECG_Dataset(X_train_1, Y_train)
    val_dataset = ECG_Dataset(X_val_1, Y_val)
    test_dataset = ECG_Dataset(X_test_1, Y_test)
    
    return train_dataset, val_dataset, test_dataset