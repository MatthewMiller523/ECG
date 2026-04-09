# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 00:00:16 2026

@author: Matthew Miller

training loader
because woo

"""


from ecg_lib.load_data_0 import load_data0
from ecg_lib.fold_func_0 import fold_0
#import settings
from pathlib import Path
import tomllib

#get settings.toml information
base_dir = Path(__file__).resolve().parent
settings_path = base_dir.parent / "settings.toml"
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

import torch
import torchvision
#import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
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

#uncomment if config is included in inputs
#config = inputs["config"]
#===============================

training_loader = torch.utils.data.DataLoader(X_train, batch_size=4, shuffle=True, num_workers=4)
validation_loader = torch.utils.data.DataLoader(X_val, batch_size=4, shuffle=False, num_workers=4)


#classes =  Y["diagnostic_superclass"].unique()