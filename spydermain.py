#main.py
'''
3/25/2026
Matthew Miller

ECG_project0/
updated from dirtymain_1 4/22/2026
'''

from ecg_lib.load_data_0 import load_data_0 as ld
from ecg_lib.fold_func_0 import fold_0
#from ecg_lib.preprocessing_1 import preprocessing_fun_0 as ppm
import sys

#import settings
from pathlib import Path
import tomllib

def main():
    #import settings
    from pathlib import Path
    import tomllib
    
    #get settings.toml information
    base_dir = Path(__file__).resolve().parent
    settings_path = base_dir / "settings.toml"
    with open(settings_path, "rb") as f:
        config = tomllib.load(f)

    #load data
    data, truth = ld(config)

    #split data
    fold_inputs = {
        'X': data,
        'Y': truth,
        'config': config
        }

    fold_data = fold_0(fold_inputs)

    #train_dataset, val_dataset, test_dataset = ppm(fold_data, config)
    
    return {
    'truth':truth
    
    }

if __name__ == "__main__":
    db = main()