#loader wrapper

#import settings
from pathlib import Path
import tomllib

import pandas as pd
#import wfdb  #unneeded as yet
import ast

import loader_func_0 as lf


#get settings.toml information
base_dir = Path(__file__).resolve().parent
settings_path = base_dir.parent / "settings.toml"

with open(settings_path, "rb") as f:
    config = tomllib.load(f)

#get path to data
data_path = config["data"]["path"]
samplingRate = config["data"]["sampling_rate"]

#load data determined by settings.toml

# load and convert annotation data
Y = pd.read_csv(data_path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = lf.load_data_0(Y, config)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(data_path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(
    lambda x: lf.aggregate_diagnostic(x, agg_df)
)    
