#loader

#imports
import wfdb
import ast

#create a loading function
#dataFrame is the input data, specifically the list of information pulled from the csv
#config is the general config of the project, passed from main.py
#sampling rate is either 100 or 500, depending on which set of data is of interest
#path is the path to the data. The filename from dataFrame.filename_## will be appended to the path
#to get the file location
def load_data_0(dataFrame, config):
    pathIn = config["data"]["path"]
    samplingRate = config["data"]["sampling_rate"]
    if samplingRate == 100:
        data = [wfdb.rdsamp(pathIn+f) for f in dataFrame.filename_lr]
    else:
        data = [wfdb.rdsamp(pathIn+f) for f in dataFrame.filename_hr]
    return data

#perform diagnostics
#y_dic is a dictionary input of diagnostic codes
#tmp is a list, created empty, filled in the if-statement
#the keys loop runs through all keys in the y_dic dictionary
#tmp.append adds the values of all elements in the y_dic dictionary to the list
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key, 'diagnostic_class'])
    return list(set(tmp))
    
    