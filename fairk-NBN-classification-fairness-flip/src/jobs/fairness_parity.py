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
    nth_length_of_sorted_lists, TrainerKey, Neighbor
from src.utils.preprocess_utils import encode_dataframe, split_df, get_xy, backward_regression, flip_value


class FairnessParity:
    def __init__(self, config: Config):
        self.val_neighbors = []
        self.config = config
        self.experiment_name = self.config.experiment_name
        self.local_dir_res = 'data/' + self.experiment_name + '/' + self.config.data.results_path
        self.local_dir_plt = 'data/' + self.experiment_name + '/' + self.config.data.plot_path
        self.sensitive_attribute = self.config.data.sensitive_attribute.name
        self.class_attribute = self.config.data.class_attribute.name

        self.sensitive_class_value = None
        self.dominant_class_value = None
        self.class_positive_value = None
        self.class_negative_value = None
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

        self.sensitive_class_value = int(self.label_encoders[self.config.data.sensitive_attribute.name].transform([self.config.data.sensitive_attribute.protected])[0])
        self.dominant_class_value = next(value for value in [0, 1] if value != self.sensitive_class_value)
        self.class_positive_value = self.config.data.class_attribute.positive_value
        self.class_negative_value = [value for value in df[self.config.data.class_attribute.name].unique() if value != self.class_positive_value][0]

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

        self.pred_val = self.model.predict(self.x_val)

        self.t0, self.to_ids = get_negative_protected_values(self.pred_val, self.x_val, self.y_val_sensitive_attr,self.class_negative_value, self.sensitive_class_value)
        self.t1, self.t1_ids = get_negative_protected_values(self.pred_val, self.x_val, self.y_val_sensitive_attr,self.class_negative_value, self.dominant_class_value)

        self.sum_positive_pred_dom = len(get_negative_protected_values(self.pred_val, self.x_val, self.y_val_sensitive_attr,self.class_positive_value, self.dominant_class_value)[0])
        self.sum_positive_pred_protected = len(get_negative_protected_values(self.pred_val, self.x_val, self.y_val_sensitive_attr,self.class_positive_value, self.sensitive_class_value)[0])
        self.sum_positive_val_dom = len(get_negative_protected_values(self.y_val, self.x_val, self.y_val_sensitive_attr, None,self.dominant_class_value)[0])
        self.sum_positive_val_protected = len(get_negative_protected_values(self.y_val, self.x_val, self.y_val_sensitive_attr, None,self.sensitive_class_value)[0])



    def _eval(self,x_data,y_data, y_sensitive_attribute,k=None):
        y_pred = self.model.predict(x_data)
        eval_results = dict()
        eval = Evaluate(y_actual=y_data,
                        y_pred= y_pred,
                        y_sensitive_attribute=y_sensitive_attribute,
                        class_attribute = self.config.data.class_attribute.name,
                        sensitive_class_value = self.sensitive_class_value,
                        dominant_class_value = self.dominant_class_value,
                        class_positive_value = self.class_positive_value,
                        class_negative_value = self.class_negative_value,
                        )

        eval_results['number_sensitive_attr_predicted_positive'] = eval.number_sensitive_attr_predicted_positive
        eval_results['pos_t0'] = (self.model.predict(self.t0) == self.class_positive_value).sum()
        eval_results['number_dom_attr_predicted_positive'] = eval.number_dom_attr_predicted_positive
        eval_results['number_indices_flipped'] = eval_results['pos_t0']
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
            t0 = get_negative_protected_values(self.y_val, self.x_val, self.y_val_sensitive_attr,self.class_negative_value, self.sensitive_class_value)
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
                if vallue == self.class_negative_value:
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
                                                                                               self.config.data.class_attribute.name) != self.class_positive_value:
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
                if value == self.class_negative_value:
                    new_sublist = sublist[:i+1]
                else:
                    new_sublist = sublist[:i]
                    break

            negative_neighbors.append(new_sublist)
        return negative_neighbors

# concise
    def _get_flip_counter(self,train_neighbors):
        counter = 0
        for neighbor in train_neighbors:
            if check_column_value(self.y_train, neighbor, self.config.data.class_attribute.name) == self.class_negative_value:
                counter += 1
        return counter


    def label_drop_attempt1(self):
        pred_val = self.model.predict(self.x_val)
        self.t0 = get_negative_protected_values(pred_val, self.x_val, self.y_val_sensitive_attr,self.class_negative_value, self.sensitive_class_value)
        k = 0
        pos_t0 = (self.model.predict(self.t0 ) == self.class_positive_value).sum()
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


            pos_t0_ar = (self.model.predict(self.t0 ) == self.class_positive_value)
            not_in_pos_t0_indices = [num for num in np.where(pos_t0_ar)[0] if num not in pos_t0_indices]
            for index_ in not_in_pos_t0_indices:
                row_df = self.t0.iloc[[index_]]
                extra_indices_removed = self._get_negative_neighbors_1by1_approach(row_df)
                indices_removed.extend(extra_indices_removed)
            pos_t0_indices.extend(not_in_pos_t0_indices)

            pos_t0_ar = (self.model.predict(self.t0) == self.class_positive_value)
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

    def filter_out_t1(self,train_neighbors):


        distance, index = self.model.kneighbors(self.t1)

        filtered_t1 = [(lst, self.t1_ids[idx]) for idx, lst in enumerate(index)
                         if any(num in train_neighbors for num in lst)]
        dom_attr_indexes, t1_ids = zip(*filtered_t1)
        return dom_attr_indexes , t1_ids

    def fill_dict_t1(self,reverse_index ,dom_attr_indexes, t1_ids,sensitive_attribute  ):
        for neighbor_id, sublist in enumerate(dom_attr_indexes):
            # Initialize a Neighbor object
            neighbor = Neighbor(
                index=t1_ids[neighbor_id],
                counter_for_flip= self._get_flip_counter(sublist),
                train_neighbors=sublist,
                sensitive_attribute= sensitive_attribute ,
                kneighbors=self.config.basic.neighbors,
                sensitive_class_value = self.sensitive_class_value,
                dominant_class_value = self.dominant_class_value,
                class_positive_value = self.class_positive_value,
                class_negative_value = self.class_negative_value
            )
            self.val_neighbors.append(neighbor)
            # Add this Neighbor to all relevant NeighborContainers
            for key in sublist:
                if key in reverse_index:
                    reverse_index[key].add_neighbor(neighbor)
        return reverse_index

    def get_reverse_index(self):

        _, index = self.model.kneighbors(self.t0)
        reverse_index =self.initialize_train_dictionary(index,self.to_ids)
        return  reverse_index



    def lebel_flip_attempt1(self):
        results_df = pd.DataFrame()
        reverse_index = self.get_reverse_index()
        reverse_index = dict(sorted(reverse_index.items(), key=lambda item: len(item[1]), reverse=True))
        #TODO : reverse index need to be nore dynamic
        #TODO : create a custom int class for the val-points or find an already implemented one that matches to the requirements

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

    def initialize_train_dictionary(self,indexes,ids):
        unique_set = set()
        for lst in indexes:
            unique_set.update(lst)
        train_neighbors = list(unique_set)
        dom_attr_indexes, t1_ids = self.filter_out_t1(train_neighbors)



        dictionary = {}
        for container_key in train_neighbors:
            if check_column_value(self.y_train, container_key, self.config.data.class_attribute.name) == 2:
                container = TrainerKey(container_key,sensitive_class_value=self.sensitive_class_value,
                dominant_class_value=self.dominant_class_value,
                class_positive_value=self.class_positive_value,
                class_negative_value=self.class_negative_value,
                include_dominant_attribute= self.config.basic.weight.include_dominant_attribute)  # Create a new NeighborContainer object
                dictionary[container_key] = container

        for neighbor_id, sublist in enumerate(indexes):
            # Initialize a Neighbor object
            neighbor = Neighbor(
                index=ids[neighbor_id],
                counter_for_flip= self._get_flip_counter(sublist),
                train_neighbors=sublist,
                sensitive_attribute= self.sensitive_class_value ,
                kneighbors=self.config.basic.neighbors,
                sensitive_class_value=self.sensitive_class_value,
                dominant_class_value=self.dominant_class_value,
                class_positive_value=self.class_positive_value,
                class_negative_value=self.class_negative_value,

            )
            self.val_neighbors.append(neighbor)
            # Add this Neighbor to all relevant NeighborContainers
            for key in sublist:
                if key in dictionary:
                    dictionary[key].add_neighbor(neighbor)
        dictionary =self.fill_dict_t1(dictionary,dom_attr_indexes, t1_ids,self.dominant_class_value)
        return dictionary

    def make_train_label_postitive(self,TrainerKey):
        for neighbor in self.reverse_index[TrainerKey].neighbors :
            if neighbor.counter_for_flip == 1:
                pass
            else:
                neighbor.counter_for_flip -= 1
        self.reverse_index.pop(TrainerKey, None)  # None prevents KeyError if key doesn't exist

    def get_difference_objective(self, pred_sensitive_flipped=None, pred_dom_flipped = None):

        self.sum_positive_pred_dom += pred_dom_flipped or 0
        self.sum_positive_pred_protected += pred_sensitive_flipped or 0

        diff = self.sum_positive_pred_protected/self.sum_positive_val_protected - self.sum_positive_pred_dom/self.sum_positive_val_dom


        return diff

    def _get_current_flips(self):
        prev_protected = self.total_protected_positive_flipped
        prev_dom    = self.total_dom_positive_flipped
        self.total_protected_positive_flipped, self.total_dom_positive_flipped = self.get_flipped_positive_counter()
        curr_protected = self.total_protected_positive_flipped - prev_protected
        curr_dom = self.total_dom_positive_flipped - prev_dom
        return curr_protected, curr_dom

    def update_results_dict(self, number_sensitive_attr_predicted_positive=None, number_sensitive_attr_predicted_negative=None,
                            number_dom_attr_predicted_positive=None, number_sensitive_attributes_flipped=None ,sum_sa_indices_flipped=None, iteration=None):
        eval_results = dict()
        eval_results['number_sensitive_attr_predicted_positive'] = number_sensitive_attr_predicted_positive
        eval_results['number_sensitive_attr_predicted_negative'] = number_sensitive_attr_predicted_negative
        eval_results['number_dom_attr_predicted_positive'] = number_dom_attr_predicted_positive
        eval_results['number_sensitive_attributes_flipped'] = number_sensitive_attributes_flipped
        eval_results['sum_sa_indices_flipped'] = sum_sa_indices_flipped
        eval_results['train_val_flipped'] = iteration
        return eval_results

    def label_flip(self):
        #TODO: get results df
        results_df = pd.DataFrame()
        train_indexer = []
        self.reverse_index = self.get_reverse_index()
        self.total_protected_positive_flipped, self.total_dom_positive_flipped = self.get_flipped_positive_counter() #should be 0
        eval_results = self.update_results_dict(number_sensitive_attr_predicted_positive =self.sum_positive_pred_protected,
                                             number_sensitive_attr_predicted_negative = len(self.t0),
                                             number_dom_attr_predicted_positive = self.sum_positive_pred_dom,
                                             number_sensitive_attributes_flipped = self.total_protected_positive_flipped,
                                             sum_sa_indices_flipped = self.total_protected_positive_flipped,
                                             iteration = len(train_indexer)+1)

        summm = 0
        diff = self.get_difference_objective()
        while(self._objective_checker()):
            objective_key = self._get_weighted_key()
            train_indexer.append(objective_key)

            self.make_train_label_postitive(objective_key)

            curr_protected_flips, curr_dom_flips = self._get_current_flips()
            diff = self.get_difference_objective(curr_protected_flips,curr_dom_flips)
            eval_results = self.update_results_dict(
                number_sensitive_attr_predicted_positive=self.sum_positive_pred_protected  ,
                number_sensitive_attr_predicted_negative=len(self.t0)- self.total_protected_positive_flipped ,
                number_dom_attr_predicted_positive= self.sum_positive_pred_dom + self.total_dom_positive_flipped,
                number_sensitive_attributes_flipped=self.total_protected_positive_flipped,
                sum_sa_indices_flipped=curr_protected_flips,
                iteration=len(train_indexer))
            results_df = results_df._append(eval_results, ignore_index=True)
        results_df.to_csv(self.local_dir_res + 'most_common_flip_results.csv', index=False)
        return results_df , train_indexer

    def _get_weighted_key(self):
        if self.config.basic.weight.second_weight:
            max_weight = float('-inf')
            keys_with_max_weight = []

            for k, item in self.reverse_index.items():
                if item.weight > max_weight:
                    max_weight, keys_with_max_weight = item.weight, [item]
                elif item.weight == max_weight:
                    keys_with_max_weight.append(item)

            result_item = min(keys_with_max_weight, key=lambda k: item.secondary_weight).index
        else:
            result_item = max(self.reverse_index, key=lambda k: self.reverse_index[k].weight)
        return result_item


    def _objective_checker(self):
        if self.config.basic.condition.affirmative_action:
            allowed_difference = (self.sum_positive_pred_dom * self.config.basic.condition.difference_percentage) // 100
            return self.sum_positive_pred_dom - self.sum_positive_pred_protected >= allowed_difference
        else:
            threshold = self.config.basic.condition.difference_percentage / 100
            diff = self.get_difference_objective()
            return abs(diff) >= threshold






    def get_flipped_positive_counter(self):
        total_protected_positive = 0
        total_dom_positive = 0
        for neighbor in self.val_neighbors:
            if neighbor.predicted_label == self.class_positive_value:
                if neighbor.sensitive_attribute == self.sensitive_class_value:
                    total_protected_positive += 1
                elif neighbor.sensitive_attribute == self.dominant_class_value:
                    total_dom_positive += 1
        return total_protected_positive, total_dom_positive

    def run_fairness_par(self):

        dataloader = DataLoader()
        df = dataloader.load_data(config=self.config)

        self._preprocess(df)
        self._train()
        reslts_df , train_indexer =self.label_flip()
        self.y_train = flip_value(self.y_train,train_indexer,self.class_positive_value,self.config.data.class_attribute.name)
        self._train() #Retrain
        test = self._eval(self.x_val, self.y_val, self.y_val_sensitive_attr, k=None) #Test
        k=1



