# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:04:51 2026

@author: Matthew Miller

Dirty main

Runs main.py but with more variables in workspace for easier debugging
"""
#=======================================
#Working and already fixed
#=======================================
from ecg_lib.load_data_0 import load_data_0 as ld
from ecg_lib.fold_func_0 import fold_0
from ecg_lib.preprocessing_1 import preprocessing_fun_0 as ppm
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

    train_dataset, val_dataset, test_dataset = ppm(fold_data, config)
    
    #============================================
    #new code begins below--4/22/2026
    #============================================

    import torch

    import torch.nn as nn
    import torch.optim as optim

    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data import Dataset, DataLoader

    from ecg_lib.models.m_conv_0 import conv_class_0 as cc_0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=True,
        num_workers=config['model']['num_workers']
        )

    '''    
    x_db, y_db = train_loader.dataset[0:5]
    print(type(y_db))
    print(y_db.shape)
    print(y_db.dtype)
    print(y_db.ndim)
    print(f"y_db, {y_db[0:5]}")
    
    print(f"type(train_loader) {type(train_loader)}")
    print(f"len(train_loader.dataset) {len(train_loader.dataset)}")
    print(f"len(train_loader) {len(train_loader)}")
    print(f"type first element x {type(x_db)}")
    print(f"shape first element x {x_db.shape}")
    print(f"type first element y {type(y_db)}")
    print(f"value first element y {y_db}")

    X_dbg, Y_dbg = next(iter(train_loader))
    print(f"shape first batch X {X_dbg.shape}")
    print(f"shape first batch Y {Y_dbg.shape}")
    print(f"shape first observation from batch {X_dbg[0].shape}")
    print(f"first label from batch {Y_dbg[0]}")
    '''
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=True,
        num_workers=config['model']['num_workers']
        )

    #I don't think I'll need this, but create it now for completeness
    #also, if I do need it later, I will never remember whether I skipped it now
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=True,
        num_workers=config['model']['num_workers']
        )
    #============================================================
    model = cc_0(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],     #x2 for bidirectional LSTM is included in classifier
        num_layers=config['model']['num_layers'],       #I think this is a vestigial relic of an earlier idea
        num_classes = config['model']['num_classes']
        )
    #============================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    #============================================================
    #loss and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = config['model']['initial_learn_rate'])

    #=============================================================
    #training loop

    val_flag = False    #val_flag starts false and goes true when the val condition hits
                        #i.e. when val_patience exceeds the val_count, switch val_flag to true
                        #can also do this directly, but that's harder to read
                        
    epoch_count = 0                    
    best_val_loss = float('inf')
    best_val_count = 0

    while epoch_count < config['model']['epochs'] and not val_flag:
        
        for epoch in range(config['model']['validation_frequency']):
            model.train()
            running_loss=0.0

            #training loop
            
            for X_batch, Y_batch in train_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.long().to(device)
                #print(f"X_batch shape before model {X_batch.shape}")
                #print(f"Y_batch shape before model {Y_batch.shape}")
                #print(f"single observation shape from batch {X_batch[0].shape}")
                
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                epoch_count += 1        #inelegant but easy to read

            #training stats to screen

            avg_loss = running_loss / len(train_loader)
            print(f"Epochs {epoch+1}/{config['model']['validation_frequency']}, Loss: {avg_loss:.4f}")
            print(f"Iterations {len(train_loader)}")    
            
            #validation loop

            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.long().to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)

                val_loss = loss.item()

                if best_val_loss > val_loss:        #e.g. we're getting better/we just hit a new best val
                    best_val_loss = val_loss
                    best_val_count = 0
                else:                           #e.g. we're not getting better/previous val was better
                    best_val_count += 1

                #validation stats to screen
                print(f"Validation Count {best_val_count}/{config['model']['validation_patience']}, Loss: {avg_loss:.4f}")

        if best_val_count >= config['model']['validation_patience']:    #val patience test
            val_flag = True

    return {                        #debug out
    'train_loader':train_loader,
    'config':config,
    'X_batch':X_batch
    }

if __name__== '__main__':
    dbg = main()