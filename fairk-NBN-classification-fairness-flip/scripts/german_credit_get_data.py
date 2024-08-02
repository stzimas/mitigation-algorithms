import os
import sys
from ucimlrepo import fetch_ucirepo 
import pandas as pd
'''
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
'''

df = pd.read_csv("data/german_credit_data_no_target.csv") #from source's keggle account 
statlog_german_credit_data = fetch_ucirepo(id=144) 
y = statlog_german_credit_data.data.targets  #1: good , 2:bad

german_credit_with_target_df = pd.concat([df, y], axis=1)
german_credit_with_target_df = german_credit_with_target_df.loc[:,~german_credit_with_target_df.columns.str.contains('^Unnamed')]

german_credit_with_target_df.to_csv("data/german_credit_wtarget.csv")