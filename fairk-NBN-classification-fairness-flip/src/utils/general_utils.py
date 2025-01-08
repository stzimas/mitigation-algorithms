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

def get_negative_protected_values(y_val,x_val,sensitive_attr_list,class_negative_value,sensitive_class_value):
    y_val = y_val.to_numpy().ravel() if isinstance(y_val, pd.DataFrame) else y_val
    if class_negative_value == None:
        mask_y_val = np.ones_like(y_val, dtype=bool)
    else:
        mask_y_val = y_val == class_negative_value
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

def categorize_and_split_sublists(sublists):
    # Determine the length of sublists based on the first sublist
    sublist_length = len(sublists[0]) if sublists else 0

    # Initialize dictionary for categories dynamically
    result = {i: [] for i in range(sublist_length + 1)}

    # Track which sublists have already been categorized
    categorized = [False] * len(sublists)

    # Compare each pair of sublists only once
    for i in range(len(sublists)):
        if categorized[i]:  # Skip if already categorized
            continue

        sublist1 = set(sublists[i])
        matched = False

        # Check for matches with other sublists
        for j in range(i + 1, len(sublists)):
            if categorized[j]:
                continue

            sublist2 = set(sublists[j])
            shared_elements = sublist1.intersection(sublist2)

            # Count the number of shared elements
            shared_count = len(shared_elements)
            if shared_count == sublist_length:
                # All elements are shared; no differences
                result[sublist_length].append([list(shared_elements), np.nan])
                categorized[i] = True
                categorized[j] = True
                matched = True
            elif 1 <= shared_count < sublist_length:
                # Some elements are shared, find the different ones
                diff1 = list(sublist1 - shared_elements)
                diff2 = list(sublist2 - shared_elements)
                result[shared_count].append([list(shared_elements), diff1 + diff2])
                categorized[i] = True
                categorized[j] = True
                matched = True

        # If no matches were found, categorize as 0
        if not matched:
            result[0].append(sublists[i])
            categorized[i] = True

    return result


def nth_length_of_sorted_lists(lists, n):
    sorted_lists = sorted(lists, key=len)
    return len(sorted_lists[n])

class Neighbor:
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
    def __init__(self, index=None, sensitive_class_value=None, dominant_class_value=None, class_positive_value=None,
                 class_negative_value=None,include_dominant_attribute=False):
        self.sensitive_class_value = sensitive_class_value
        self.dominant_class_value = dominant_class_value
        self.class_positive_value = class_positive_value
        self.class_negative_value = class_negative_value
        self.include_dominant_attribute = include_dominant_attribute
        self.index = index
        self.neighbors = []

    @property
    def weight(self):
        sum= 0
        for neighbor in self.neighbors:
            if neighbor.sensitive_attribute == self.sensitive_class_value:
                sum += 1/ neighbor.counter_for_flip
            elif neighbor.sensitive_attribute == self.dominant_class_value and  self.include_dominant_attribute:
                sum += -1/ neighbor.counter_for_flip

        return sum

    @property
    def secondary_weight(self):
        sum = 0
        for neighbor in self.neighbors:
            if neighbor.sensitive_attribute == self.dominant_class_value:
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

#TODO: sum_positive_dom_attr
    # List to hold references to Neighbor objects

    def add_neighbor(self, neighbor):
        """Add a Neighbor to the container."""
        self.neighbors.append(neighbor)

    def __iter__(self):
        """Make the container iterable."""
        return iter(self.neighbors)

    def __getitem__(self, index):
        """Allow access by index."""
        return self.neighbors[index]

    def __len__(self):
        """Return the number of neighbors."""
        return len(self.neighbors)

    def __repr__(self):
        """Provide a string representation of the container."""
        return f"Train_Neighbor({self.index})"


