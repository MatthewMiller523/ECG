# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:46:24 2026

@author: Matthew Miller

#fp
"""
#general first
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler as StanScal
from sklearn.preprocessing import MultiLabelBinarizer

#mine second
from .ECG_Dataset_1 import ECG_Dataset, ECG_md_Dataset

#====================================================
#junk

def md_shape_array_1(inputs, runtime):       #shapes the inputs into a tensor for the CNN
    num_samples = shape(inputs)
    num_features = runtime.md_features_number

    out = np.zeros((num_samples, num_features))

    for sample_index in range(num_samples):
        
        col_sample = inputs[sample_index]
        col_sample = col_sample.fillna(col_sample.median())
        
        out[sample_index, 0, :] = col_sample
            #inputs[sample_index]

    print(f"out.shape {out[0,0].shape}")
    return out
#=======================================================

def d_average_slice_0(inputs):
    
#inputs = data[0][0]
    
    len_streams, num_streams = inputs.shape
    out_primitive=np.zeros((1, len_streams))
    
    for sample in range(len_streams):
        out_primitive[0,sample] = inputs[sample,:].sum()/num_streams
        
    return out_primitive    
    
def d_average_array_0(inputs, cfg):
    num_samples = len(inputs)
    out = np.zeros((num_samples, 1, 1000, cfg.data['num_leads']))
    #print(f"out.shape {out.shape}")
    #print(f"len(inputs) {len(inputs)}")
    #print(f"type(inputs) {type(inputs)}")
    #print(f"len(inputs[0]) {len(inputs[0])}")
    #print(f"type(inputs[0]) {type(inputs[0])}")
    #print(f"inputs[0][0].shape {inputs[0][0].shape}")

    for sample_index in range(num_samples):
        for lead_index in range(cfg.data['num_leads']):
            out[sample_index, 0] = inputs[sample_index][0][lead_index]

    #print(f"out.shape {out[0,0].shape}")
    return out

def preprocessing_func_0(inputs, cfg):
    #unpack inputs
    X_train = inputs['X_train']
    X_val = inputs['X_val']
    X_test = inputs['X_test']
    Y_train = inputs['Y_train']
    Y_val = inputs['Y_val']
    Y_test = inputs['Y_test']

    #============================

    X_train_0 = d_average_array_0(X_train, cfg)
    X_val_0 = d_average_array_0(X_val, cfg)
    X_test_0 = d_average_array_0(X_test, cfg)

    X_train_1 = torch.tensor(X_train_0, dtype=torch.float32)
    X_val_1 = torch.tensor(X_val_0, dtype=torch.float32)
    X_test_1 = torch.tensor(X_test_0, dtype=torch.float32)

    train_dataset = ECG_Dataset(X_train_1, Y_train)
    val_dataset = ECG_Dataset(X_val_1, Y_val)
    test_dataset = ECG_Dataset(X_test_1, Y_test)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.model['batch_size'],
        shuffle=True,
        num_workers=cfg.model['num_workers']
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.model['batch_size'],
        shuffle=True,
        num_workers=cfg.model['num_workers']
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.model['batch_size'],
        shuffle=True,
        num_workers=cfg.model['num_workers']
        )
    
    return train_loader, val_loader, test_loader
    
#============================================================
#============================================================    


def d_shape_array_1(inputs, cfg):       #shapes the inputs into a tensor for the CNN
    num_samples = len(inputs)
    num_leads = cfg.data['num_leads']
    sample_width = 10 * cfg.data['sampling_rate']
    
    out = np.zeros((num_samples, 1, sample_width, num_leads))
    
    #print(f"out.shape {out.shape}")
    #print(f"len(inputs) {len(inputs)}")
    #print(f"type(inputs) {type(inputs)}")
    #print(f"len(inputs[0]) {len(inputs[0])}")
    #print(f"type(inputs[0]) {type(inputs[0])}")
    #print(f"inputs[0][0].shape {inputs[0][0].shape}")

    for sample_index in range(num_samples):
        out[sample_index, 0, :, :] = \
            inputs[sample_index][0]

    #print(f"out.shape {out[0,0].shape}")
    return out

#z-score normilization 
def norm_func_0(inputs, runtime):

    inputs=inputs.copy()
    sample_length = runtime.sample_length
    
    train = inputs['train']
    val = inputs['val']
    test = inputs['test']

    scaler_p = [] #this is the parameter storage object for the scalers

    #the default implementation, this function, is to do all three, train, val, and test in one go. That's because I don't want to be carrying around norm constants
    #recall data looks like (N, 1000/5000, 12)
    #so I will need to do multiple slices.
    #each slice should be (N, 1000/5000) for comprehensible clarity, meaning I need 12 slices per t/v/t
    train_norm = np.zeros((runtime.num_train, 1, sample_length, 12))
    val_norm = np.zeros((runtime.num_val, 1, sample_length, 12))
    test_norm = np.zeros((runtime.num_test, 1, sample_length, 12))        

    for lead_index in range(12):
        scaler= StanScal()  #there will be one scaler for each lead
        
        train_norm[:,0,:,lead_index] = scaler.fit_transform(train[:,0,:,lead_index])
        val_norm[:,0,:,lead_index] = scaler.transform(val[:,0,:,lead_index])
        test_norm[:,0,:,lead_index] = scaler.transform(test[:,0,:,lead_index])

        scaler_p.append(scaler)

    #train_norm = scaler.fit_transform(train_slice)
    #val_norm = scaler.transform(val_slice)
    #test_norm = scaler.transform(test_slice)

    output = {
    'train_norm' : train_norm,
    'val_norm' : val_norm,
    'test_norm' : test_norm
    }
    
    return output

def md_shape_array_2(inputs, runtime):

    inputs = inputs.copy()
    inputs = inputs.fillna(inputs.median(numeric_only=True))

    out = inputs.to_numpy(dtype=np.float32)

    runtime.md_features_number = out.shape[1]

    return out
    
def preprocessing_func_1(inputs, cfg, runtime):
    #unpack inputs
    X_train = inputs['X_train']
    X_val = inputs['X_val']
    X_test = inputs['X_test']
    Y_train = inputs['Y_train']
    Y_val = inputs['Y_val']
    Y_test = inputs['Y_test']
    md_train = inputs['md_train']
    md_val = inputs['md_val']
    md_test = inputs['md_test']
    #============================

    X_train_0 = d_shape_array_1(X_train, cfg)
    X_val_0 = d_shape_array_1(X_val, cfg)
    X_test_0 = d_shape_array_1(X_test, cfg)

    runtime.num_train = X_train_0.shape[0]
    runtime.num_val = X_val_0.shape[0]
    runtime.num_test = X_test_0.shape[0]

    md_train_0 = md_shape_array_2(md_train, runtime)
    md_val_0 = md_shape_array_2(md_val, runtime)
    md_test_0 = md_shape_array_2(md_test, runtime)

    X_train_1 = torch.tensor(X_train_0, dtype=torch.float32)
    X_val_1 = torch.tensor(X_val_0, dtype=torch.float32)
    X_test_1 = torch.tensor(X_test_0, dtype=torch.float32)

    X_to_norm = {
        'train': X_train_1,
        'val': X_val_1,
        'test': X_test_1
        }

    X_norm = norm_func_0(X_to_norm, runtime)
    
    X_train_norm = X_norm['train_norm']
    X_val_norm = X_norm['val_norm']
    X_test_norm = X_norm['test_norm']

    md_train_1 = torch.tensor(md_train_0, dtype=torch.float32)
    md_val_1 = torch.tensor(md_val_0, dtype=torch.float32)
    md_test_1 = torch.tensor(md_test_0, dtype=torch.float32)

    #normalization for metadata    
    scaler_md = StanScal()
    md_train_norm = scaler_md.fit_transform(md_train_1)
    md_val_norm = scaler_md.transform(md_val_1)
    md_test_norm = scaler_md.transform(md_test_1)

    #store this because it might be useful later
    runtime.md_mean = scaler_md.mean_
    runtime.md_sigma = scaler_md.scale_

    mlb = MultiLabelBinarizer()
    
    Y_train_0 = mlb.fit_transform(Y_train)
    Y_val_0 = mlb.transform(Y_val)
    Y_test_0 = mlb.transform(Y_test)
    
    runtime.label_classes = mlb.classes_
    runtime.num_classes = len(mlb.classes_)

    train_dataset = ECG_md_Dataset(X_train_norm, Y_train_0, md_train_norm)
    val_dataset = ECG_md_Dataset(X_val_norm, Y_val_0, md_val_norm)
    test_dataset = ECG_md_Dataset(X_test_norm, Y_test_0, md_test_norm)

    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.model['batch_size'],
        shuffle=True,
        num_workers=cfg.model['num_workers']
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.model['batch_size'],
        shuffle=True,
        num_workers=cfg.model['num_workers']
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.model['batch_size'],
        shuffle=True,
        num_workers=cfg.model['num_workers']
        )

    return train_loader, val_loader, test_loader 