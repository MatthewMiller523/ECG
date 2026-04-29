# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:04:51 2026

@author: Matthew Miller

N=0.1
"""
#=======================================
#Working and already fixed
#=======================================
from ecg_lib.read_config_0 import Config
from ecg_lib.load_data_1 import load_data_0 as ld
from ecg_lib.fold_func_0 import fold_0 as fold
from ecg_lib.preprocessing_2 import preprocessing_fun_0 as ppm
from ecg_lib.train_1 import m_train as mtrain
from ecg_lib.model_test_0 import m_test_0 as mtest
from ecg_lib.output_write_0 import o_write_0 as o_write
import sys


def main():

    cfg = Config()

    #load data
    data, truth = ld(cfg)
    fold_data = fold(data, truth, cfg)
    train_loader, val_loader, test_loader = ppm(fold_data, cfg)

    #train model
    t_model, status_var = mtrain(train_loader, val_loader, cfg)

    #write training metadata to file
    if cfg.meta['output_csv']:
        o_write(status_var, cfg)
    
    #test model
    t_loss, t_acc = mtest(test_loader, t_model)


    return status_var

if __name__== '__main__':
    dbg = main()        #dbg is all debugging (dbg) variables