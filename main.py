# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:04:51 2026

@author: Matthew Miller

N=0.1
"""
#=======================================
#Working and already fixed
#=======================================
from ecg_lib.load_data_0 import load_data_0 as ld
from ecg_lib.fold_func_0 import fold_0
from ecg_lib.preprocessing_1 import preprocessing_fun_0 as ppm
from ecg_lib.train_0 import m_train as mtrain_0
from ecg_lib.model_test_0 import m_test_0 as mtest_0 
import sys

#import settings
from pathlib import Path
import tomllib

def main():

    #get settings.toml information
    base_dir = Path(__file__).resolve().parent
    settings_path = base_dir / 'settings.toml'
    with open(settings_path, 'rb') as f:
        config = tomllib.load(f)

    #load data
    data, truth = ld(config)

    #split data
    fold_inputs = {
        'X': data,
        'Y': truth,
        'config': config
        }

    fold_data = fold_0(fold_inputs)
    train_loader, val_loader, test_loader = ppm(fold_data, config)
    
    #train_inputs = {
    #    'train_loader':train_loader,
    #    'val_loader':train_loader
    #    }
        
    t_model = mtrain_0(train_loader, val_loader, config)    #for trained model

    t_loss, t_acc = mtest_0(test_loader, t_model)

if __name__== '__main__':
    dbg = main()