# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:13:13 2026

@author: Matthew Miller

version0, purely loads data from the PTB-XL directory as indicated
outputs are X and Y, wherein X is signal data in np.arrays and Y is a dataframe

making things!

This is the load_data0 function, to be called from main.py. This file should be in project0/ecg_lib/
wherein project0 is the current home directory.

config is a dict, ripped from the settings.toml file by main.py

loader_func_0.py should be colocated with this file

I'm getting pushing into blah_blah naming conventions instead of blahBlah :(

"""

#import main packages
from pathlib import Path
import tomllib
import pandas as pd
import ast

#import functions
from . import loader_func_0 as lf

def load_data0(config):
    try:
        data_path = config["data"]["path"]
        sampling_rate = config["data"]["sampling_rate"]
    except KeyError as e:
        raise KeyError(f"Missing required config key: {e}") from e
        
    if sampling_rate not in (100, 500):
        raise ValueError(
            f"Unsupported sampling_rate {sampling_rate!r} in settings.toml. Expected 100 or 500.")
        
    data_dir = Path(data_path)
    ptbxl_csv = data_dir / "ptbxl_database.csv"  #this is the csv of files and their sampling rates
    scp_csv = data_dir / "scp_statements.csv"    
    
    #SCP-ECG is the Standard Communications Protoccol for computer assisted electrocardiography
    #there's a wikipedia page: https://en.wikipedia.org/wiki/SCP-ECG
    #it's the patient notes


    #does everything exist?        
    if not ptbxl_csv.is_file():
        raise FileNotFoundError(f"PTB-XL csv database not found: {ptbxl_csv}")
    if not scp_csv.is_file():
        raise FileNotFoundError(f"SCP_CSV statements not found: {scp_csv}")

    #opens datafiles truth
    try:
        Y = pd.read_csv(ptbxl_csv, index_col="ecg_id")
    except Exception as e:
        raise RuntimeError(f"Failed to read PTB-XL annotations from {ptbxl_csv}") from e
        
    #creates truth (y) with literals from scp        
    try:
        Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
    except Exception as e:
        raise RuntimeError(f"Failed to parse Y.scp_codes with ast.literal_eval") from e
        #I don't really know what this means

    #loads datafiles signal        
    try:
        X = lf.load_data_0(Y,config)
    except Exception as e:
        raise RuntimeError(f"Failed while loading raw wfdb signal data") from e

    #processes scp notes    
    try:
        agg_df = pd.read_csv(scp_csv, index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]
    except Exception as e:
        raise RuntimeError(f"Failed to read or filter scp statements from {scp_csv}") from e

    #adds diagnosis to truth        
    try:
        Y["diagnostic_superclass"] = Y.scp_codes.apply(
            lambda x: lf.aggregate_diagnostic(x, agg_df)
            )
    except Exception as e:
        raise RuntimeError(f"Failed to aggregate diagnostic superclass") from e
        
    return [X, Y]

    
    
