# -*- coding: utf-8 -*-
#=============================
#train and val

import torch
import copy

import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
#from torch.utils.data import Dataset, DataLoader

from ecg_lib.models.m_conv_1 import (
    conv_class_0 as cc_0,
    conv_class_1 as cc_1,
    conv_class_2 as cc_2
    )

def m_train(train_loader, val_loader, cfg):

    #train_loader = inputs['train_loader']
    #val_loader = inputs['val_loader']
    
    
    #============================================
    #new code begins below--4/22/2026
    #============================================
    model = cc_1(
        input_size=cfg.model['input_size'],
        hidden_size=cfg.model['hidden_size'],     #x2 for bidirectional LSTM is included in classifier
        num_layers=cfg.model['num_layers'],       #I think this is a vestigial relic of an earlier idea
        num_classes = cfg.model['num_classes']
        )
    #============================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    #============================================================
    #loss and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = cfg.model['initial_learn_rate'])

    #=============================================================
    #training loop

    val_flag = False    #val_flag starts false and goes true when the val condition hits
                        #i.e. when val_patience exceeds the val_count, switch val_flag to true
                        #can also do this directly, but that's harder to read
                        
    epoch_count = 0                    
    best_val_loss = float('inf')
    best_val_count = 0
    #print(device)
    #print(torch.cuda.is_available())
    #print(torch.cuda.get_device_name(0))
    headers = ['epochs','train loss','validation count','val loss','best val loss']
    status_dict = {'0':headers}
    status_count = 1
    
    for epoch in range(cfg.model['epochs']):
        if val_flag:    
            break           #make this a return command once this is moved into a subfunction
            
        model.train()
        running_loss=0.0

        #training loop
        batch_count = 0
        
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
            
            batch_count += 1        #inelegant but easy to read
            val_loss = 0
            if batch_count % cfg.model['validation_frequency'] == 0:

                avg_loss = running_loss / len(train_loader)
                
                #training stats to screen
                print(f"Epochs {epoch + 1}/{cfg.model['epochs']}, Loss: {avg_loss:.4f}")
                #print(f"Iterations {len(train_loader)}")
                epoch_nums = [(epoch+1),avg_loss]     

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
                    #torch.save(model.state_dict(), 'best_model.pt')
                    best_model = copy.deepcopy(model.state_dict())
                else:                           #e.g. we're not getting better/previous val was better
                    best_val_count += 1

                #validation stats to screen
                print(f"Validation Count {best_val_count}/{cfg.model['validation_patience']}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}")
                val_numbs = [best_val_count, val_loss, best_val_loss]
                
                status_nums = epoch_nums + val_numbs
                status_dict[status_count] = status_nums
                status_count += 1
                
                if best_val_count >= cfg.model['validation_patience']:    #val patience test
                    val_flag = True
                    break

    status_df = pd.DataFrame(status_dict).T
                
    model.load_state_dict(best_model)

    return model, status_df
    #{                        #debug out
    #'train_loader':train_loader,
    #'config':config,
    #'X_batch':X_batch,
    #'best_model':best_model
    #}


