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


def d_average_array_0(inputs, config):
    num_samples = len(inputs)
    out = np.zeros((num_samples, 1, 1000, config["data"]["num_leads"]))
    #print(f"out.shape {out.shape}")
    #print(f"len(inputs) {len(inputs)}")
    #print(f"type(inputs) {type(inputs)}")
    #print(f"len(inputs[0]) {len(inputs[0])}")
    #print(f"type(inputs[0]) {type(inputs[0])}")
    #print(f"inputs[0][0].shape {inputs[0][0].shape}")

    for sample_index in range(num_samples):
        for lead_index in range(config["data"]["num_leads"]):
            out[sample_index, 0] = inputs[sample_index][0][lead_index]

    #print(f"out.shape {out[0,0].shape}")
    return out
