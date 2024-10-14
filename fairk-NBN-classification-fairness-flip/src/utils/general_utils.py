import logging
import sys

import numpy as np
import pandas as pd
from tomlkit import value


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


def check_column_value(df, index, column_name):
    """
    Check the value of a specified column at a given index in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to check.
    index (int): The index of the row to check.
    column_name (str): The name of the column to check.

    Returns:
    The value of the specified column at the given index, or an error message if the index or column is out of range.
    """
    if index in df.index:
        if column_name in df.columns:
            value = df.at[index, column_name]
            return value
        else:
            return "Column name does not exist."
    else:
        return "Index out of range."

def group_sublists_by_shared_elements(n, list_of_lists):
    # Step 1: Filter out sublists with the specified length 'n'
    filtered_sublists = [lst for lst in list_of_lists if len(lst) == n]

    # Step 2: Create empty groups dynamically based on the length 'n'
    groups = [[] for _ in range(n + 1)]  # n+1 groups for 0 to n shared elements

    # Step 3: Compare each sublist with others to group by shared elements
    while filtered_sublists:
        current = filtered_sublists.pop(0)  # Take the first sublist

        # Temporary lists for the current sublist to track grouped elements
        current_groups = [[] for _ in range(n + 1)]
        current_groups[n].append(current)  # Start by placing in "all shared" group

        # Compare current sublist with remaining ones
        for sublist in filtered_sublists[:]:
            shared_elements = len(set(current) & set(sublist))  # Count shared elements
            current_groups[shared_elements].append(sublist)  # Group by shared count
            filtered_sublists.remove(sublist)  # Remove after grouping

        # Add the grouped sublists into the final groups
        for i in range(n + 1):
            if current_groups[i]:
                groups[i].append(current_groups[i])

    return groups

def filter_sublists_by_length(lst, length):
    return [sublist for sublist in lst if len(sublist) == length]





