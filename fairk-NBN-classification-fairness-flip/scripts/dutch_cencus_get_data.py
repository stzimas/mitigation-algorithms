import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.io.arff import loadarff



def load(arff_file, decode_str=True):

    data, meta = loadarff(arff_file)
    df = DataFrame(data, columns=meta.names())
    if decode_str:
        df_str = df.select_dtypes(include=['object'])
        if not df_str.empty:
            df[df_str.columns] = df_str.map(lambda x: x.decode('utf-8'))
    return df
#https://github.com/tailequy/fairness_dataset/tree/main/Dutch_census

filepath = "../data/dutch_census_2001.arff"
class_label = {'5_4_9':2,'2_1':1}

df = load(filepath)
df = df.rename(columns={'occupation': 'class'})
sex_counts = df['sex'].value_counts() #sex_counts:('2',30273)('1', 30147) #by the surveys standard
protected_class_label = {'2':'female','1':'male'}
df['class'] = df['class'].map(class_label)
df['sex'] = df['sex'].map(protected_class_label)
df = df.rename(columns={'sex': 'Sex'})

df.to_csv('../data/dutch_cencus_wtarget.csv', index=False)

k=1
