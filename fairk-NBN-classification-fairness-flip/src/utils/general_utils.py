import logging
import sys

import numpy as np
import pandas as pd


def _innit_logger(name="logs.logs"):
    with open(name, "w") as log_file:
        pass
    logging.root.handlers = []
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO,
                        handlers=[logging.FileHandler(name),logging.StreamHandler(sys.stdout)])

def get_neighbor_statistics(index, df, target_column):
    try:
        if isinstance(index, (list, np.ndarray)):
            values = df.loc[index, target_column].tolist()
        else:
            values = df.loc[index, target_column]
        return values
    except KeyError:
        return "The specified index or column does not exist in the DataFrame."

def get_negative_protected_values(y_val,x_val,sensitive_attr_list,class_attribute):
    mask_y_val = y_val[class_attribute] == 2
    mask_y_val_sensitive_attr = pd.Series(sensitive_attr_list) == 0
    combined_mask = mask_y_val & mask_y_val_sensitive_attr
    t0 = x_val[combined_mask]
    return t0




