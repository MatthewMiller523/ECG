# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:38:03 2026

@author: Matthew Miller

fold and split, 
v0 just folds and splits
v1 separates metadata for ml-stack injection

put broad runtime information derivations at bottom of fold_1 or later

"""

import random
#import numpy as np

#starts in fold_1
from ecg_lib.md_compile_0 import mdc_0 as mdc

def fold_0(X, Y, cfg):

    #unpack inputs if necessary
    #X = inputs['X']
    #Y = inputs['Y']
    #config = inputs['config']
    train_index = cfg.data['train_index']
    #train_key = config.data['train_key']

    #random.seed(config["general"]["r_seed"]            #set random module seed
    #np.random.seed(config["general"]["np_r_seed"])     #set np seed
    test_fold = cfg.model['test_fold']                  #the test_fold is a constant testdata set
    val_fold = 9                                        #just use random for now/could set this to 9
    #val_fold = config["model"]["validation_fold"]      #set validation fold


    val_mask = (Y.strat_fold == val_fold)               #selected out val and test
    test_mask = (Y.strat_fold == test_fold)
    train_mask = ~(test_mask | val_mask)                #anything left is train

    X_train = [x for x, keep in zip(X, train_mask.values) if keep]
    X_val = [x for x, keep in zip(X, val_mask.values) if keep]
    X_test = [x for x, keep in zip(X, test_mask.values) if keep]
    
    Y_train = Y[train_mask].iloc[:,train_index]
    Y_val = Y[val_mask].iloc[:,train_index]
    Y_test = Y[test_mask].iloc[:,train_index]
    
    outputs = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'Y_train': Y_train,
        'Y_val': Y_val,
        'Y_test': Y_test
        }

    return outputs
    
    
def fold_1(X, Y, cfg, runtime):

    train_index = cfg.data['train_index']
    train_class = cfg.data['train_key']

    #random.seed(config["general"]["r_seed"]            #set random module seed
    #np.random.seed(config["general"]["np_r_seed"])     #set np seed
    test_fold = cfg.model['test_fold']                  #the test_fold is a constant testdata set
    val_fold = 9                                        #just use random for now/could set this to 9
    #val_fold = config["model"]["validation_fold"]      #set validation fold

    val_mask = (Y.strat_fold == val_fold)               #selected out val and test
    test_mask = (Y.strat_fold == test_fold)
    train_mask = ~(test_mask | val_mask)                #anything left is training data

    X_train = [x for x, keep in zip(X, train_mask.values) if keep]
    X_val = [x for x, keep in zip(X, val_mask.values) if keep]
    X_test = [x for x, keep in zip(X, test_mask.values) if keep]
    
    Y_train = Y[train_mask].loc[:, train_class]
    Y_val = Y[val_mask].loc[:, train_class]
    Y_test = Y[test_mask].loc[:, train_class]
    
    md_train = Y[train_mask].drop(columns = [train_class])
    md_val = Y[val_mask].drop(columns = [train_class])
    md_test = Y[test_mask].drop(columns = [train_class])
 
    md_data, runtime = mdc(md_train, md_val, md_test, cfg, runtime)
    
    md_train = md_data['out_train']
    md_val = md_data['out_val']
    md_test = md_data['out_test']
    
    runtime.sample_length = cfg.data['sampling_rate']*10

    outputs = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'Y_train': Y_train,
        'Y_val': Y_val,
        'Y_test': Y_test,
        'md_train': md_train,   #figure out which of these is better and use that
        'md_val': md_val,
        'md_test': md_test
        }

    return outputs, runtime