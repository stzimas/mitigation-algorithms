import logging
import os
import shutil
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import omegaconf
import sys

from sklearn.neighbors import KNeighborsClassifier


from src.dataloader.dataloader import DataLoader
from src.config.schema import Config
from src.jobs.evaluate import Evaluate
from src.utils.general_utils import get_neighbor_statistics, get_negative_protected_values, check_column_value, \
    group_sublists_by_shared_elements, filter_sublists_by_length, categorize_and_split_sublists, \
    nth_length_of_sorted_lists
from src.utils.preprocess_utils import encode_dataframe, split_df, get_xy, backward_regression, flip_value


class FairnessParity:
    def __init__(self, config: Config):
        self.config = config
        self.experiment_name = self.config.experiment_name
        self.local_dir_res = 'data/' + self.experiment_name + '/' + self.config.data.results_path
        self.local_dir_plt = 'data/' + self.experiment_name + '/' + self.config.data.plot_path
        self.sensitive_attribute = self.config.data.sensitive_attribute.name
        self.class_attribute = self.config.data.class_attribute.name
        self.categorical_attribute = None
        self.label_encoders = {}
        self.features = None
        paths = [self.local_dir_res, self.local_dir_plt]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    for dir in dirs:
                        shutil.rmtree(os.path.join(root, dir))

    def _preprocess(self, df):
        df, self.label_encoders = encode_dataframe(df)
        if self.config.basic.feature_selection:
            df = backward_regression(df, self.sensitive_attribute)

        self.features = df.columns.values.tolist()
        self.features.remove(self.class_attribute)

        train_set, val_set, test_set = split_df(df=df,
                                                split_percent=self.config.basic.split_percent,
                                                protected_attribute=self.config.data.sensitive_attribute.name,
                                                val_data=self.config.basic.split_data.val_data,
                                                resampling_train_set=self.config.basic.split_data[
                                                    'resampling_train_set'], )

        self.x_train, self.y_train, self.y_train_sensitive_attr = get_xy(df=train_set,
                                                        sensitive_attribute=self.config.data.sensitive_attribute.name,
                                                        target_column=self.config.data.class_attribute.name)
        self.x_test, self.y_test, self.y_test_sensitive_attr = get_xy(df=test_set,
                                                       sensitive_attribute=self.config.data.sensitive_attribute.name,
                                                       target_column=self.config.data.class_attribute.name)
        self.x_val, self.y_val, self.y_val_sensitive_attr = get_xy(df=val_set,
                                                    sensitive_attribute=self.config.data.sensitive_attribute.name,
                                                    target_column=self.config.data.class_attribute.name)

        if self.config.basic.exclude_sensitive_attribute:
            self.x_train = self.x_train.drop(self.sensitive_attribute,axis=1)
            self.x_val = self.x_val.drop(self.sensitive_attribute,axis=1)
            self.x_test = self.x_test.drop(self.sensitive_attribute,axis=1)

        return self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test, self.y_train_sensitive_attr, self.y_val_sensitive_attr, self.y_test_sensitive_attr


    def _train(self,sns=None):

        self.model = KNeighborsClassifier(n_neighbors=self.config.basic.neighbors)
        self.model.fit(self.x_train, self.y_train[self.class_attribute])

    def _eval(self,k):
        y_pred = self.model.predict(self.x_test)
        eval_results = dict()
        eval = Evaluate(y_actual=self.y_test, y_pred= y_pred, y_sensitive_attribute=self.y_sensitive_attr,
                        class_attribute=self.class_attribute)

        eval_results['number_sensitive_attr_predicted_positive'] = eval.number_sensitive_attr_predicted_positive
        eval_results['pos_t0'] = (self.model.predict(self.t0) == 1).sum()
        eval_results['number_dom_attr_predicted_positive'] = eval.number_dom_attr_predicted_positive
        eval_results['number_indices_flipped'] = eval_results['pos_t0']
        eval_results['sum_indices_flipped'] = eval_results['pos_t0'] - self.number_indices_flipped_bef
        eval_results['iteration'] = k


        return eval_results
    def remove_indices_from_train(self, indices):
        if not isinstance(indices, list):
            indices = [indices]
        indices = list(set(indices))

        y_train_sensitive_attr_series = pd.Series(self.y_train_sensitive_attr, index=self.x_train.index)
        combined_df = pd.concat([self.x_train, self.y_train, y_train_sensitive_attr_series.rename('sensitive_attr')],
                                axis=1)

        combined_df = combined_df.drop(index=indices, errors="ignore")
        combined_df = combined_df.reset_index(drop=True)

        self.x_train = combined_df.iloc[:, :-2]
        self.y_train = combined_df.iloc[:, self.x_train.shape[1]:self.x_train.shape[1] + 1]
        self.y_train_sensitive_attr = combined_df.iloc[:, -1].tolist()

    def _get_negative_neighbors_1by1_approach(self, negative_classified_protected_class=None,invert_back_to_orignal = False):
        if negative_classified_protected_class is None:
            t0 = get_negative_protected_values(self.y_val, self.x_val, self.y_val_sensitive_attr, self.class_attribute)
        else:
            t0 = negative_classified_protected_class
        original_xtrain = self.x_train.copy()
        original_ytrain = self.y_train.copy()
        original_y_train_sensitive_attr = self.y_train_sensitive_attr.copy()
        t0_negative_neighbors = []
        for i in range(len(t0)):
            neighbors = []
            original_index = 0
            while (True):
                distance, index = self.model.kneighbors(t0.iloc[[i]])
                index_to_be_removed = index[0].tolist()[0]
                vallue = check_column_value(self.y_train, index_to_be_removed, self.config.data.class_attribute.name)
                if vallue == 2:
                    neighbors.append(index_to_be_removed + original_index)
                    self.remove_indices_from_train(index_to_be_removed)
                    self._train()
                else:
                    same_distance_removal = self._get_neighbors_with_same_distance(t0.iloc[[i]],index_to_be_removed)
                    self.remove_indices_from_train(same_distance_removal)
                    t0_negative_neighbors.extend(same_distance_removal)
                    self._train()

                    if invert_back_to_orignal == True:
                        self.x_train = original_xtrain
                        self.y_train = original_ytrain
                        self.y_train_sensitive_attr = original_y_train_sensitive_attr
                        self._train()

                    t0_negative_neighbors.extend(neighbors)
                    break
        return t0_negative_neighbors

    def _get_neighbors_with_same_distance(self,negative_classified_protected_class,neighbor_index):
        distance, index = self.model.kneighbors(negative_classified_protected_class, n_neighbors=350)
        index_position = np.where(index[0] == neighbor_index)[0]
        target_distance = distance[0][index_position]
        same_distance_removal = []
        for i, distance in zip(index[0], distance[0]):
            if i != neighbor_index and distance == target_distance and check_column_value(self.y_train, i,
                                                                                               self.config.data.class_attribute.name) != 1:
                same_distance_removal.append(i)
        return same_distance_removal

    def _get_negative_classified_protected_class_subset(self,df,n_neighbors=384):
        distances, indices = self.model.kneighbors(df, n_neighbors=n_neighbors,return_distance=True)
        flattened_array = distances.flatten()

        unique_values = np.unique(flattened_array)

        total_elements = flattened_array.size
        unique_count = unique_values.size
        percentage_unique = (unique_count / total_elements) * 100
        negative_neighbors = []
        for sublist in indices:
            new_sublist = []
            for i,element in enumerate(sublist):
                value = check_column_value(self.y_train,element,self.config.data.class_attribute.name)
                if value == 2:
                    new_sublist = sublist[:i+1]
                else:
                    new_sublist = sublist[:i]
                    break

            negative_neighbors.append(new_sublist)
        return negative_neighbors

# concise

    def label_drop_attempt1(self):
        pred_val = self.model.predict(self.x_val)
        self.t0 = get_negative_protected_values(pred_val, self.x_val, self.y_val_sensitive_attr, self.class_attribute)
        k = 0
        pos_t0 = (self.model.predict(self.t0 ) == 1).sum()
        print(f"positive_t0: {pos_t0}")
        sum_idices_flipped = 0
        results_df = pd.DataFrame()
        pos_t0_indices = []
        og_xtrain_len = len(self.x_train)
        for index, row in self.t0.iterrows():
            results_dict = dict()

            row_df = self.t0 .iloc[[k]]
            '''
            if k == 4:
                row_df = self.t0.iloc[[1]]
            here = (self.model.predict(row_df))
            if here[0] == 1:
                print("hey")
            '''
            indices_removed = self._get_negative_neighbors_1by1_approach(row_df)
            y_pred_val = self.model.predict(self.x_val)


            pos_t0_ar = (self.model.predict(self.t0 ) == 1)
            not_in_pos_t0_indices = [num for num in np.where(pos_t0_ar)[0] if num not in pos_t0_indices]
            for index_ in not_in_pos_t0_indices:
                row_df = self.t0.iloc[[index_]]
                extra_indices_removed = self._get_negative_neighbors_1by1_approach(row_df)
                indices_removed.extend(extra_indices_removed)
            pos_t0_indices.extend(not_in_pos_t0_indices)

            pos_t0_ar = (self.model.predict(self.t0) == 1)
            pos_t0 = pos_t0_ar.sum()
            eval = Evaluate(y_actual=self.y_val, y_pred=y_pred_val, y_sensitive_attribute=self.y_val_sensitive_attr,
                            class_attribute=self.class_attribute)
            number_sensitive_attr_predicted_positive = eval.number_sensitive_attr_predicted_positive
            number_dom_attr_predicted_positive = eval.number_dom_attr_predicted_positive
            results_dict['number_sensitive_attr_predicted_positive'] = number_sensitive_attr_predicted_positive
            results_dict['pos_t0'] = pos_t0
            results_dict['number_dom_attr_predicted_positive'] = number_dom_attr_predicted_positive
            results_dict['number_indices_removed'] = og_xtrain_len - len(self.x_train)
            results_dict['sum_indices_removed'] = len(indices_removed)
            results_dict['iteration'] = k
            results_df = results_df._append(results_dict, ignore_index=True)
            print(results_dict)

            if number_sensitive_attr_predicted_positive >= number_dom_attr_predicted_positive:
                results_df.to_csv(self.local_dir_res + 'label_drop_attempt1.csv', index=False)
                return results_df

            print(k)
            k = k + 1

    def get_reverse_index(self):
        pred_val = self.model.predict(self.x_val)
        self.t0 = get_negative_protected_values(pred_val, self.x_val, self.y_val_sensitive_attr, self.class_attribute)

        reverse_index = defaultdict(list)

        distance, index = self.model.kneighbors(self.t0)
        for i, num_list in enumerate(index):
            # Extract the single element from each list, assuming it's always a single-item list
            num = num_list[0]
            reverse_index[num].append(i)



        return reverse_index


    def lebel_flip_attempt1(self):
        results_df = pd.DataFrame()
        reverse_index = self.get_reverse_index()
        reverse_index = dict(sorted(reverse_index.items(), key=lambda item: len(item[1]), reverse=True))
        #TODO : reverse index need to be nore dynamic
        k=0
        results_dict = self._eval(k)
        results_df = results_df._append(results_dict, ignore_index=True)

        k=k+1
        for key, value in reverse_index.items():
            self.y_train.loc[key, self.class_attribute] = 1  # flip label
            self._train()

            results_dict = self._eval(k)
            results_df = results_df._append(results_dict, ignore_index=True)
            k+=1
            if results_dict['number_sensitive_attr_predicted_positive'] >= results_dict['number_dom_attr_predicted_positive']:
                results_df.to_csv(self.local_dir_res + 'most_common_flip_results.csv', index=False)
                break


    def run_fairness_par(self):

        dataloader = DataLoader()
        df = dataloader.load_data(config=self.config)

        self._preprocess(df)
        self._train()
        #TODO: logic for experiment results




