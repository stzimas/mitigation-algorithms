import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm


def split_df(df, split_percent, protected_attribute,class_attribute, val_data=False,resampling_train_set=True,sensitive_class_value = None, dominant_class_value = None, class_positive_value = None, class_negative_value = None):
    """
    Splits a DataFrame into training, testing, and optionally validation sets while maintaining
    balance across a protected attribute.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    split_percent (float): The percentage of data to allocate to the test set.
    protected_attribute (str): The name of the protected attribute column.
    val_data (bool, optional): Whether to create a validation set from the test set. Defaults to False.
    resampling_train_set (bool): Whether to balance the training set based on the protected attribute. Defaults to True.
    sensitive_class_value (any): The value of the protected attribute representing the sensitive group.
    dominant_class_value (any): The value of the protected attribute representing the dominant group.
    class_positive_value (any): The value of the target class considered positive.
    class_negative_value (any): The value of the target class considered negative.

    Returns:
        - pd.DataFrame: The training set.
        - pd.DataFrame or None: The validation set (if `val_data` is True, otherwise None).
        - pd.DataFrame: The test set.
    """
    def generate_groups(df):
        """Creates a dictionary of data subsets based on class and protected attribute values."""
        return {
            'dom_attr_positive': df[(df[class_attribute] == class_positive_value) & (df[protected_attribute] == dominant_class_value)],
            'sen_attr_positive': df[(df[class_attribute] == class_positive_value) & (df[protected_attribute] == sensitive_class_value)],
            'dom_attr_negative': df[(df[class_attribute] == class_negative_value) & (df[protected_attribute] == dominant_class_value)],
            'sen_attr_negative': df[(df[class_attribute] == class_negative_value) & (df[protected_attribute] == sensitive_class_value)]
        }
    def get_samples(df,split_percent):
        """Selects a stratified sample from the dataset based on the split percentage."""
        n_samples_per_group = int(len(df) * split_percent / 4)
        groups = generate_groups(df)
        test_samples = []
        for group_name, group_df in groups.items():
            sampled_group = group_df.sample(n=n_samples_per_group, random_state=42)
            test_samples.append(sampled_group)
        test_set = pd.concat(test_samples)
        return test_set

    test_set = get_samples(df, split_percent)

    train_set = df.drop(test_set.index)
    test_set = test_set.reset_index(drop=True).sample(frac=1, random_state=42).reset_index(drop=True)
    train_set = train_set.reset_index(drop=True).sample(frac=1, random_state=42).reset_index(drop=True)
    if resampling_train_set:
        positive_train_set = train_set[train_set[class_attribute] == class_positive_value]
        negative_train_set = train_set[train_set[class_attribute] == class_negative_value]

        min_count1 = min(positive_train_set[protected_attribute].value_counts())
        min_count2 = min(negative_train_set[protected_attribute].value_counts())

        balanced_positive_train_set = pd.concat([
            positive_train_set[positive_train_set[protected_attribute] == dominant_class_value].sample(min_count1 - (min_count1//2) , random_state=1),
            positive_train_set[positive_train_set[protected_attribute] == sensitive_class_value].sample(min_count1, random_state=1)
        ])

        balanced_negative_train_set = pd.concat([
            negative_train_set[negative_train_set[protected_attribute] == dominant_class_value].sample(min_count2, random_state=1),
            negative_train_set[negative_train_set[protected_attribute] == sensitive_class_value].sample(min_count2-(min_count2//2), random_state=1)
        ])

        train_set = pd.concat([balanced_positive_train_set, balanced_negative_train_set]).reset_index(drop=True)

    groups = generate_groups(df)

    for group_name, group_df in groups.items():
        print(f"{group_name}: {len(group_df)} samples")



    if val_data:
        val_set = get_samples(test_set, 0.5)
        test_set = test_set.drop(val_set.index)
        test_set = test_set.reset_index(drop=True).sample(frac=1, random_state=42).reset_index(drop=True)
        val_set = val_set.reset_index(drop=True).sample(frac=1, random_state=42).reset_index(drop=True)
        return train_set, val_set, test_set
    else:
        return train_set,None, test_set.reset_index(drop=True)


def treat_categorical_data():
    pass


def encode_dataframe(df):
    """
        Encodes categorical columns in a DataFrame using Label Encoding.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing categorical columns.

        Returns:
            - pd.DataFrame: The transformed DataFrame with categorical columns encoded as integers.
            - dict: A dictionary mapping column names to their respective LabelEncoder instances.
        """
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df , label_encoders

def decode_dataframe(encoded_df, encoders):
    decoded_df = encoded_df.copy()
    for column, le in encoders.items():
        decoded_df[column] = le.inverse_transform(encoded_df[column])
    return decoded_df


def get_xy(df, sensitive_attribute ,target_column):
    """
        Splits a DataFrame into features (X), target (y), and a list of sensitive attribute values.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        sensitive_attribute (str): The name of the column representing the sensitive attribute.
        target_column (str): The name of the target column.

        Returns:
            - pd.DataFrame: The feature set (X) with the target column removed.
            - pd.DataFrame: The target variable (y) as a DataFrame.
            - list: A list of values from the sensitive attribute column.
        """

    x = df.drop(columns=[target_column])
    y = df[[target_column]]
    y_sensitive_attr = df[sensitive_attribute].tolist()
    return x, y, y_sensitive_attr


def backward_regression(df,sensitive_attribute, threshold_out=0.01):
    """
        Perform Backward Stepwise Regression to select features for modeling.

        This function iteratively removes features from the DataFrame `df` based on their p-values
        obtained from an Ordinary Least Squares (OLS) model. The process continues until no features
        with p-values exceeding the specified `threshold_out` are left.

        Args:
            df (DataFrame): The input DataFrame containing features and target variable.
            threshold_out (float): The p-value threshold used for feature removal. Features with
                                   p-values greater than this threshold will be removed.

        Returns:
            DataFrame: The modified DataFrame with selected features for modeling.
    """
    selected_features = []
    x = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    included = list(x.columns)
    while True:

        changed = False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        '''
        max_tuple = max(pvalues, key=lambda item: item[1])
        if max_tuple[0] == sensitive_attribute:
            pvalues.remove(max_tuple)
        '''
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            logging.info(f"worst_feature : {worst_feature}, {worst_pval} ")
        if not changed:
            break
    selected_features.append(included)
    logging.info(f"\nSelected Features:\n{selected_features[0]}")
    if sensitive_attribute  in selected_features[0]:
        selected_features[0].remove(sensitive_attribute)
    x_train = df[[sensitive_attribute]+selected_features[0]]
    y_train = df.iloc[:, -1]
    train_data = pd.concat([x_train, y_train], axis=1)
    return train_data


def flip_value(df, indices, new_value, column_name=None):
    """
      Flips the values of specified indices in a given DataFrame column.

      Args:
          df (pd.DataFrame): The input DataFrame.
          indices (int, list, or np.int64): The index or list of indices to modify.
          new_value (any): The new value to assign to the specified indices.
          column_name (str, optional): The column in which values should be modified.

      Returns:
          pd.DataFrame: A copy of the DataFrame with the updated values.
      """
    df=df.copy()
    if isinstance(indices, int) or isinstance(indices, np.int64):
        indices = [indices]
    for index in indices:
        df.loc[index, column_name] = new_value
    return df

'''
empty_sublists_count = sum(1 for sublist in snegative_clasified_protected_class_subset if len(sublist) == 0)
        pos_t0 = (self.model.predict(t0) == 1).sum()
        results = []
        for i in range(len(snegative_clasified_protected_class_subset)):
            sublist = snegative_clasified_protected_class_subset[i]
            number_to_check = indices1[i][0]
            exists = number_to_check in sublist or len(sublist) == 0
            results.append(exists)
        false_count = results.count(False)
'''