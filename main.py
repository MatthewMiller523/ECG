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
from ecg_lib.fold_func_1 import fold_1 as fold      #old ~fold_func_0 import fold_0 as fold
from ecg_lib.preprocessing_2 import preprocessing_func_1 as ppm
from ecg_lib.train_2 import m_train as mtrain
from ecg_lib.model_test_2 import m_test_1 as mtest
from ecg_lib.output_write_0 import o_write_0 as o_write
from ecg_lib.runtime_config import runtime_config_class as rcc
import sys


def main():

    cfg = Config()
    runtime = rcc()

    #load data
    data, truth = ld(cfg)
    fold_data, runtime = fold(data, truth, cfg, runtime)

    print("X_train:", len(fold_data["X_train"]))
    print("X_val:", len(fold_data["X_val"]))
    print("X_test:", len(fold_data["X_test"]))

    print("Y_train:", len(fold_data["Y_train"]))
    print("Y_val:", len(fold_data["Y_val"]))
    print("Y_test:", len(fold_data["Y_test"]))

    print("md_train:", len(fold_data["md_train"]))
    print("md_val:", len(fold_data["md_val"]))
    print("md_test:", len(fold_data["md_test"]))

    train_loader, val_loader, test_loader = ppm(fold_data, cfg, runtime)

    #train model
    t_model, status_var = mtrain(train_loader, val_loader, cfg, runtime)

    #write training metadata to file
    if cfg.meta['output_csv']:
        o_write(status_var, cfg)

    #test model
    t_loss, t_acc_label, t_acc_exact = mtest(test_loader, t_model)


    return t_loss, t_acc_label, t_acc_exact

if __name__== '__main__':
    dbg = main()        #dbg is all debugging (dbg) variables