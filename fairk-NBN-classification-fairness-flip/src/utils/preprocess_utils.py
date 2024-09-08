import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm


def split_df(df, split_percent, protected_attribute, val_data=False,resampling_train_set=True):
    def generate_groups(df):
        return {
            'z0_positive': df[(df['class'] == 1) & (df[protected_attribute] == 0)],
            'z1_positive': df[(df['class'] == 1) & (df[protected_attribute] == 1)],
            'z0_negative': df[(df['class'] == 2) & (df[protected_attribute] == 0)],
            'z1_negative': df[(df['class'] == 2) & (df[protected_attribute] == 1)]
        }
    def get_samples(df,split_percent):
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
        positive_train_set = train_set[train_set['class'] == 1]
        negative_train_set = train_set[train_set['class'] == 2]

        # Balance counts for column1 in df1
        min_count1 = min(positive_train_set[protected_attribute].value_counts())
        min_count2 = min(negative_train_set[protected_attribute].value_counts())

        balanced_positive_train_set = pd.concat([
            positive_train_set[positive_train_set[protected_attribute] == 0].sample(min_count1 - (min_count1//2) , random_state=1),
            positive_train_set[positive_train_set[protected_attribute] == 1].sample(min_count1, random_state=1)
        ])

        balanced_negative_train_set = pd.concat([
            negative_train_set[negative_train_set[protected_attribute] == 0].sample(min_count2, random_state=1),
            negative_train_set[negative_train_set[protected_attribute] == 1].sample(min_count2-(min_count2//2), random_state=1)
        ])

        train_set =  pd.DataFrame(None)
        train_set = pd.concat([balanced_positive_train_set, balanced_negative_train_set]).reset_index(drop=True)

    groups = generate_groups(train_set)

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
        worst_pval = pvalues.max()  # null if pvalues is empty
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