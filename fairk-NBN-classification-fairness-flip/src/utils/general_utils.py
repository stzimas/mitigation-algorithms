import logging
import sys

import numpy as np
import pandas as pd
from pandera import DataFrameSchema, Column, Check


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

def get_negative_protected_values(y_val,x_val,sensitive_attr_list,class_negative_value,sensitive_class_value):
    """
        Extracts samples from x_val where the target variable (y_val) corresponds to the  class
        and the sensitive attribute matches the specified sensitive class value.

        Parameters:
        y_val (pd.DataFrame, np.ndarray, or list): The target variable values.
        x_val (pd.DataFrame): The feature dataset.
        sensitive_attr_list (list): A list of sensitive attribute values corresponding to x_val.
        class_negative_value (any): The value representing the negative class.
        sensitive_class_value (any): The value representing the sensitive class.

        Returns:
            - pd.DataFrame: The subset of x_val matching the negative class and sensitive attribute.
            - pd.Index: The index values of the selected subset.
        """
    y_val = y_val.to_numpy().ravel() if isinstance(y_val, pd.DataFrame) else y_val
    if class_negative_value == None:
        mask_y_val = np.ones_like(y_val, dtype=bool)
    else:
        mask_y_val = y_val == class_negative_value
    if sensitive_class_value == None:
        mask_y_val_sensitive_attr = np.ones_like(y_val, dtype=bool)
    else:
        mask_y_val_sensitive_attr = pd.Series(sensitive_attr_list) == sensitive_class_value
    combined_mask = mask_y_val & mask_y_val_sensitive_attr
    t0_ids = x_val[combined_mask].index
    t0 = x_val[combined_mask]
    return t0 ,t0_ids


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
    filtered_sublists = [lst for lst in list_of_lists if len(lst) == n]

    groups = [[] for _ in range(n + 1)]  # n+1 groups for 0 to n shared elements

    while filtered_sublists:
        current = filtered_sublists.pop(0)  # Take the first sublist

        current_groups = [[] for _ in range(n + 1)]
        current_groups[n].append(current)  # Start by placing in "all shared" group

        for sublist in filtered_sublists[:]:
            shared_elements = len(set(current) & set(sublist))  # Count shared elements
            current_groups[shared_elements].append(sublist)  # Group by shared count
            filtered_sublists.remove(sublist)  # Remove after grouping

        for i in range(n + 1):
            if current_groups[i]:
                groups[i].append(current_groups[i])

    return groups

def filter_sublists_by_length(lst, length):
    return [sublist for sublist in lst if len(sublist) == length]

def categorize_and_split_sublists(sublists):
    sublist_length = len(sublists[0]) if sublists else 0

    result = {i: [] for i in range(sublist_length + 1)}

    categorized = [False] * len(sublists)

    for i in range(len(sublists)):
        if categorized[i]:  # Skip if already categorized
            continue

        sublist1 = set(sublists[i])
        matched = False

        for j in range(i + 1, len(sublists)):
            if categorized[j]:
                continue
            sublist2 = set(sublists[j])
            shared_elements = sublist1.intersection(sublist2)
            shared_count = len(shared_elements)
            if shared_count == sublist_length:
                result[sublist_length].append([list(shared_elements), np.nan])
                categorized[i] = True
                categorized[j] = True
                matched = True
            elif 1 <= shared_count < sublist_length:
                diff1 = list(sublist1 - shared_elements)
                diff2 = list(sublist2 - shared_elements)
                result[shared_count].append([list(shared_elements), diff1 + diff2])
                categorized[i] = True
                categorized[j] = True
                matched = True

        if not matched:  #no matches were found, categorize as 0
            result[0].append(sublists[i])
            categorized[i] = True

    return result


def nth_length_of_sorted_lists(lists, n):
    sorted_lists = sorted(lists, key=len)
    return len(sorted_lists[n])

class Neighbor:
    """
        Represents a neighbor instance in the classification model.

        Attributes:
            index (int): The unique identifier of the neighbor.
            counter_for_flip (int): The number of times this neighbor has been flipped in classification.
            sensitive_attribute (any, optional): The sensitive attribute associated with the neighbor.
            train_neighbors (list, optional): A list of neighboring training instances.
            kneighbors (int): The number of neighbors considered in classification.
            sensitive_class_value (any, optional): The value representing the sensitive class.
            dominant_class_value (any, optional): The value representing the dominant class.
            class_positive_value (any, optional): The value representing a positive classification.
            class_negative_value (any, optional): The value representing a negative classification.

        """
    def __init__(self, index, counter_for_flip, sensitive_attribute=None, train_neighbors=None,
                 kneighbors=3
                 , sensitive_class_value=None, dominant_class_value=None, class_positive_value=None,
                 class_negative_value=None):
        if train_neighbors is None:
            train_neighbors= []
        self.index = index
        self.counter_for_flip =counter_for_flip
        self.sensitive_attribute = sensitive_attribute
        self.train_neighbors = train_neighbors
        self.kneighbors = kneighbors
        self.sensitive_class_value  = sensitive_class_value
        self.dominant_class_value  = dominant_class_value
        self.class_positive_value = class_positive_value
        self.class_negative_value =class_negative_value

    @property
    def predicted_label(self):
        if self.kneighbors is None or self.kneighbors == 0:
            raise ValueError("kneighbors must be a non-zero positive integer.")
        return self.class_positive_value if self.counter_for_flip / self.kneighbors <= 0.5 else self.class_negative_value

    def __repr__(self):
        return f"Val_Neighbor({self.index})"

class TrainerKey:
    """
        Represents a key entity in the training process, storing neighbors and their attributes.

        Attributes:
            index (int, optional): The unique identifier for the trainer key.
            sensitive_class_value (any, optional): The value representing the sensitive class.
            dominant_class_value (any, optional): The value representing the dominant class.
            class_positive_value (any, optional): The value representing a positive classification.
            class_negative_value (any, optional): The value representing a negative classification.
            neighbors (list): A list of neighboring `Neighbor` objects.

        """
    def __init__(self, index=None, sensitive_class_value=None, dominant_class_value=None, class_positive_value=None,
                 class_negative_value=None):
        self.sensitive_class_value = sensitive_class_value
        self.dominant_class_value = dominant_class_value
        self.class_positive_value = class_positive_value
        self.class_negative_value = class_negative_value
        self.index = index
        self.neighbors = []

    @property
    def weight(self):
        sum= 0
        for neighbor in self.neighbors:
            if neighbor.sensitive_attribute == self.sensitive_class_value and neighbor.predicted_label != self.class_positive_value:
                sum += 1/ neighbor.counter_for_flip


        return sum

    @property
    def secondary_weight(self):
        sum = 0
        for neighbor in self.neighbors:
            if neighbor.sensitive_attribute == self.dominant_class_value and neighbor.predicted_label != self.class_positive_value:
                sum += 1 / neighbor.counter_for_flip
        return sum

    @property
    def sum_positive_protected_attr(self):
        sum = 0
        for neighbor in self.neighbors:
            if neighbor.sensitive_attribute == self.sensitive_class_value and neighbor.predicted_label == self.class_positive_value:
                sum += 1
        return sum

    @property
    def sum_positive_dom_attr(self):
        sum = 0
        for neighbor in self.neighbors:
            if neighbor.sensitive_attribute == self.dominant_class_value and neighbor.predicted_label == self.class_positive_value:
                sum += 1
        return sum

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def __iter__(self):
        return iter(self.neighbors)

    def __getitem__(self, index):
        return self.neighbors[index]

    def __len__(self):
        return len(self.neighbors)

    def __repr__(self):
        return f"train_neighbor({self.index})"


def update_results_dict(number_sensitive_attr_predicted_positive=None, number_sensitive_attr_predicted_negative=None,
                        number_dom_attr_predicted_positive=None, number_sensitive_attributes_flipped=None, number_flipped=None, sum_sa_indices_flipped=None, sum_indices_flipped=None, iteration=None,
                        rpr=None,bpr=None,diff=None):
    """
       Creates and returns a dictionary containing evaluation results.

       Args:
           number_sensitive_attr_predicted_positive (int, optional): Count of sensitive attributes predicted as positive.
           number_sensitive_attr_predicted_negative (int, optional): Count of sensitive attributes predicted as negative.
           number_dom_attr_predicted_positive (int, optional): Count of dominant attributes predicted as positive.
           number_sensitive_attributes_flipped (int, optional): Count of sensitive attributes that were flipped.
           number_flipped (int, optional): Total count of flipped labels.
           sum_sa_indices_flipped (int, optional): Sum of sensitive attribute indices that were flipped.
           sum_indices_flipped (int, optional): Sum of all indices that were flipped.
           iteration (int, optional): Current iteration number in the  process.

       Returns:
           dict: A dictionary containing evaluation metrics.
       """
    eval_results = dict()
    eval_results['number_sensitive_attr_predicted_positive'] = number_sensitive_attr_predicted_positive
    eval_results['number_sensitive_attr_predicted_negative'] = number_sensitive_attr_predicted_negative
    eval_results['number_dom_attr_predicted_positive'] = number_dom_attr_predicted_positive
    eval_results['number_sensitive_attributes_flipped'] = number_sensitive_attributes_flipped
    eval_results['number_flipped'] = number_flipped
    eval_results['sum_sa_indices_flipped'] = sum_sa_indices_flipped
    eval_results['sum_indices_flipped'] = sum_indices_flipped
    eval_results['rpr'] = rpr
    eval_results['bpr'] = bpr
    eval_results['obj_difference'] = diff
    eval_results['train_val_flipped'] = iteration
    return eval_results


def rename_columns_(df,path):
    """
        Renames specific columns in the given DataFrame and saves the modified DataFrame as a CSV file.

        Args:
            df (pd.DataFrame): The input DataFrame to be modified.
            path (str): The file path where the modified DataFrame will be saved.

        Returns:
            None
        """
    df =df.copy()
    rename_dict = {
        'number_sensitive_attr_predicted_positive': 'Red Predicted Positive',
        'number_sensitive_attr_predicted_negative': 'Red Predicted Negative',
        'number_dom_attr_predicted_positive': 'Blue Predicted Positive',
        'number_sensitive_attributes_flipped': 'Red Flipped',
        'number_flipped':'Total Flipped',
        'sum_sa_indices_flipped': 'Sum of Red Flipped',
        'sum_indices_flipped': 'Sum Flipped',
        'train_val_flipped':'Sum of Train Points Flipped',
        'rpr':'Red Positive Rate',
        'bpr':'Blue Positive Rate',
        'obj_difference':'RPR - BPR' ,


    }
    df = df.rename(columns=rename_dict)
    df.to_csv(path, index=False)

def check_data_schema(df,class_attribute,sensitive_attribute ):
    if class_attribute not in df.columns:
        raise ValueError(f"Column '{class_attribute}' does not exist in the dataset.")
    if sensitive_attribute not in df.columns:
        raise ValueError(f"Column '{sensitive_attribute}' does not exist in the dataset.")

    schema = DataFrameSchema({
        class_attribute: Column(
            dtype=df[class_attribute].dtype,
            checks=Check(lambda x: x.nunique() == 2, error="Must have exactly 2 unique values")
        ),
        sensitive_attribute: Column(
            dtype=df[sensitive_attribute].dtype,
            checks=Check(lambda x: x.nunique() == 2, error="Must have exactly 2 unique values")
        )
    })

    schema.validate(df)


def export_test_statistics(before_stats, after_stats, filename=None):
    """
    Compare two sets of test statistics and export them as a CSV file.

    :param before_stats: Dictionary containing statistics before changes
    :param after_stats: Dictionary containing statistics after changes
    :param filename: Name of the CSV file to save
    """
    data = []
    for key in before_stats:
        data.append(["Before", key, before_stats[key]])
        data.append(["After", key, after_stats[key]])

    df = pd.DataFrame(data, columns=["State", "Metric", "Value"])
    if filename is None:
        return df
    df.to_csv(filename, index=False)



