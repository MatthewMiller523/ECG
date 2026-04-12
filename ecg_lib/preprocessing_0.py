# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:46:24 2026

@author: Matthew Miller

#f'king python'
"""

import numpy as np

def d_average_slice_0(inputs):
    
#inputs = data[0][0]
    
    len_streams, num_streams = inputs.shape
    out_primitive=np.zeros((1, len_streams))
    
    for sample in range(len_streams):
        out_primitive[0,sample] = inputs[sample,:].sum()/num_streams
        
    return out_primitive    


def d_average_array_0(inputs):
    num_samples = len(inputs)
    out = np.zeros((num_samples, ))
    
    
    for sample_index in range(num_samples):
        
    
    