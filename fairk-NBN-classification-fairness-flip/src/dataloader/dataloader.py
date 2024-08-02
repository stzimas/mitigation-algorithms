import os
import pandas as pd

class DataLoader:

    def __init__(self):
        self.data_name = None
        
    def load_data(self, config):

        df = pd.read_csv(config.data.load_from)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        return df
