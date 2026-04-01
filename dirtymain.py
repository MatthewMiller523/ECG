# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:04:51 2026

@author: Matthew Miller

Dirty main

Runs main.py but with more variables in workspace for easier debugging
"""

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


#call the training function in train.py

