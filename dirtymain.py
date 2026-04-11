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
from ecg_lib.load_data_0 import load_data0
from ecg_lib.fold_func_0 import fold_0
#import settings
from pathlib import Path
import tomllib

#get settings.toml information
base_dir = Path(__file__).resolve().parent
settings_path = base_dir / "settings.toml"
with open(settings_path, "rb") as f:
    config = tomllib.load(f)

#load data
data, truth = load_data0(config)

#split data
fold_inputs = {
    "X": data,
    "Y": truth,
    "config": config
    }

fold_data = fold_0(fold_inputs)
# %%


#call the training function in train.py
#============================================
#training_loader_0.py
#============================================

import torch
import torchvision
#import torchvision.transforms as transforms


from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from ecg_lib.models.m_classifier_0 import Classifier_0

from datetime import datetime

class_selection = "diagnostic_superclass"

#def training_loader_0(inputs)
inputs = fold_data

#unpack inputs if necessary
X_train = inputs["X_train"]
X_val = inputs["X_val"]
X_test = inputs["X_test"]
Y_train = inputs["Y_train"]
Y_val = inputs["Y_val"]
Y_test = inputs["Y_test"]
# %%


#uncomment if config is included in inputs
#config = inputs["config"]
#===============================

#this is a class because I'm going to deal with datasets as an opject. Ideally, this keeps things together
#better than functions. idk

class ECG_Dataset(Dataset):
    def __init__(self, X, Y):
        '''
        X: [N, seq_len, num_features]
        Y: [N]
        '''
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.long)   #fix this when ready

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_i = self.X[idx]
        y_i = self.Y[idx]
        return x_i, y_i
        

#===========================================================
#Do data stuff
#num_features = len(input_data) This might not be constant
#num_classes = num classes      This should be constant across everything
#__everything's in the config__


#========================================================

train_dataset = ECG_Dataset(X_train, Y_train)
val_dataset = ECG_Dataset(X_val, Y_val)
test_dataset = ECG_Dataset(X_test, Y_test)

print(X_train.shape)
print(Y_train.shape)


train_loader = DataLoader(
    train_dataset,
    batch_size=config["model"]["batch_size"],
    shuffle=True,
    num_workers=config["model"]["num_workers"]
    )

val_loader = DataLoader(
    val_dataset,
    batch_size=config["model"]["batch_size"],
    shuffle=True,
    num_workers=config["model"]["num_workers"]
    )

#I don't think I'll need this, but create it now for completeness
#also, if I do need it later, I will never remember if I skipped it now
test_loader = DataLoader(
    test_dataset,
    batch_size=config["model"]["batch_size"],
    shuffle=True,
    num_workers=config["model"]["num_workers"]
    )
#============================================================
    
model = Classifier_0(
    input_size=config["model"]["input_size"],
    hidden_size=config["model"]["hidden_size"],     #x2 for bidirectional LSTM is included in classifier
    num_layers=config["model"]["num_layers"],       #I think this is a vestigial relic of an earlier idea
    num_classes = config["model"]["num_classes"]
    )

#============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#============================================================
#loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.adam(model.parameters(), lr = config["model"]["initial_learn_rate"])

#=============================================================
#training loop

val_flag = false    #val_flag starts false and goes true when the val condition hits
                    #i.e. when val_patience exceeds the val_count, switch val_flag to true
                    #can also do this directly, but that's harder to read
                    
epoch_count = 0                    
best_val_loss = float('inf')
best_val_count = 0

while epoch_count < config["model"]["epochs"] and not val_flag:
    
    for epoch in range(config["model"]["validation_frequency"]):
        model.train()
        running_loss=0.0

        #training loop
        
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            epoch_count += 1        #inelegant but easy to read

        #training stats to screen

        avg_loss = running_loss / len(train_loader)
        print(f"Epochs {epoch+1}/{config["model"]["validation_frequency"]}, Loss: {avg_loss:.4f}")
        print(f"Iterations {len(train_loader)}")    
        
        #validation loop

        for X_batch, Y_batch in val_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)

            if best_val_loss > loss:        #e.g. we're getting better/we just hit a new best val
                best_val_loss = loss
                best_val_count = 0
            else:                           #e.g. we're not getting better/previous val was better
                best_val_count += 1

            #validation stats to screen
            print(f"Validation Count {best_val_count}/{config["model"]["validation_patience"]}, Loss: {avg_loss:.4f}")

    if best_val_count >= config["model"]["validation_patience"]:    #val patience test
        val_flag = true

        