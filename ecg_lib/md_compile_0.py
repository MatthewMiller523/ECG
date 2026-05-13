# -*- coding: utf-8 -*-
"""
metadata compiler

rips metadata out of the md_train variable and stuffs it into an output that can be concatenated into the datastream in the train_1.cc_3 and similar convolution based models
"""

import pandas as pd
#from ecg_lib.runtime_config import runtime_config_class as rcc

def make_md_dataframe(df, key_list, scp_keys):

    all_keys = key_list + sorted(scp_keys)

    rows = []

    for idx, row in df.iterrows():

        out_row = {}

        # normal dataframe columns
        for key in key_list:
            out_row[key] = row[key]

        # sub-dictionary inside scp_codes
        scp_dict = row['scp_codes']

        for key in sorted(scp_keys):
            out_row[key] = scp_dict.get(key, 0)

        rows.append(out_row)

    out_df = pd.DataFrame(rows, columns=all_keys, index=df.index)

    return out_df

def mdc_0(in_train, in_val, in_test, cfg, runtime):

    #runtime = rcc()

    key_list = ['age', 'weight']

    scp_keys = set()

    for d in in_train['scp_codes']:
        scp_keys.update(d.keys())

    all_keys = key_list + sorted(scp_keys)

    runtime.md_features_number = len(all_keys)
    runtime.md_features = all_keys

    out_train = make_md_dataframe(in_train, key_list, scp_keys)
    out_val   = make_md_dataframe(in_val, key_list, scp_keys)
    out_test  = make_md_dataframe(in_test, key_list, scp_keys)

    data = {
        'out_train': out_train,
        'out_val': out_val,
        'out_test': out_test
    }

    return data, runtime


