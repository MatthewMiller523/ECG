# -*- coding: utf-8 -*-
#=============================

from pathlib import Path
import time


def o_write_0(status_var, cfg):
    
    outdir = Path('data')
    outdir.mkdir(exist_ok=True)

    unix_time = time.time()
    time_add = int(unix_time % int(1e5))    #number of digits of addendum is the exponent here e.g. 5
    short_name = cfg.meta['csv_name'] #I could do all this in one line
    filename = f"{short_name}{time_add}.csv"#but it's easier to read individually
    long_name = outdir / filename
    
    with open(long_name, mode = 'w', newline = '', encoding = 'utf-8') as f:
        status_var.to_csv(f, index=False)