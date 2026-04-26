# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:38:03 2026

@author: Matthew Miller

fold and split

"""

import random
#import numpy as np

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