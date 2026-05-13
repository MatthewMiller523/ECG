# -*- coding: utf-8 -*-
'''
Created on 5/7/2026
@author: Matthew Miller

runtime settings

config information that is empirically obtained through operations
such as: dimensions of data,
changeable model parameters (input sizes for a particular layer, etc.),
gpu usage information

'''

from dataclasses import dataclass, field

@dataclass
class runtime_config_class:
    md_features_number: int = 0               #number of metadata features
    md_features: list = field(default_factory = list)
    sample_length: int = 0
    md_mean: float = 0.0
    md_sigma: float = 0.0
    num_train: int = 0
    num_val: int = 0
    num_test: int = 0
    label_classes: list = field(default_factory = list)
    num_classes: int = 0
    input_shape: tuple = None
    
    
    