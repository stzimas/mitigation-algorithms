import logging
import os
import random
import shutil
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from src.config.schema import Config
from src.jobs.evaluate import Evaluate
from src.utils.general_utils import get_negative_protected_values, check_column_value, TrainerKey, Neighbor, \
    update_results_dict, rename_columns_, check_data_schema, export_test_statistics
from src.utils.preprocess_utils import encode_dataframe, split_df, get_xy, backward_regression, flip_value


class FairnessParity:
    def __init__(self, config: Config=None, knn_neighbors=None,
                                            class_attribute=None,
                                            sensitive_attribute=None,
                                            sensitive_attribute_protected=None,
                                            positive_class_value=None,
                                            feature_selection=False,
                                            split_percent=0.2,
                                            basic_split = False,
                                            has_val_data=True,
                                            exclude_sensitive_attribute=True,
                                            resampling_train_set= False,
                                            second_weight = True,
                                            sensitive_catches_dominant = True,
                                            random_train_point = False,
                                            affirmative_action = False,
                                            difference_percentage = 0.0 ,
                                            load_from = None,
                                            data = None,
                                            experiment_name="fairknn",
                                            return_model =False,
                                            local_dir_res="results/",
                                            local_dir_plt="plots/",
                                            csv_to_word = False):
        self.reverse_index = None
        self.t0 = None
        self.y_train_sensitive_attr = None
        self.y_train = None
        self.x_train = None
        self.total_protected_positive_flipped = None
        self.total_dom_positive_flipped = None
        self.val_neighbors = []
        if config == None:
            self.knn_neighbors = knn_neighbors
            self.experiment_name = experiment_name
            self.return_model = return_model
            self.local_dir_res = self.experiment_name + '/' + local_dir_res
            self.local_dir_plt = self.experiment_name + '/' + local_dir_plt
            self.sensitive_attribute = sensitive_attribute
            self.class_attribute = class_attribute
            self.sensitive_attribute_protected = sensitive_attribute_protected
            self.positive_class_value = positive_class_value
            self.feature_selection = feature_selection
            self.split_percent = split_percent
            self.basic_split = basic_split
            self.random_train_point = random_train_point
            self.has_val_data = has_val_data
            self.resampling_train_set =resampling_train_set
            self.exclude_sensitive_attribute = exclude_sensitive_attribute
            self.second_weight = second_weight
            self.sensitive_catches_dominant = sensitive_catches_dominant
            self.affirmative_action = affirmative_action
            self.difference_percentage = difference_percentage
            self.load_from = load_from
            self.csv_to_word = csv_to_word
        else:
            self.config = config
            self.knn_neighbors = self.config.basic.neighbors
            self.experiment_name = self.config.experiment_name
            self.local_dir_res = 'data/' + self.experiment_name + '/' + self.config.data.results_path
            self.local_dir_plt = 'data/' + self.experiment_name + '/' + self.config.data.plot_path
            self.sensitive_attribute = self.config.data.sensitive_attribute.name
            self.class_attribute = self.config.data.class_attribute.name
            self.sensitive_attribute_protected = self.config.data.sensitive_attribute.protected
            self.positive_class_value = self.config.data.class_attribute.positive_value
            self.feature_selection = self.config.basic.feature_selection
            self.split_percent = self.config.basic.split_percent
            self.basic_split = self.config.basic.basic_split
            self.random_train_point = self.config.basic.random_train_point
            self.has_val_data = self.config.basic.split_data.val_data
            self.resampling_train_set = self.config.basic.split_data.resampling_train_set
            self.exclude_sensitive_attribute = self.config.basic.exclude_sensitive_attribute
            self.second_weight = self.config.basic.weight.second_weight
            self.sensitive_catches_dominant = self.config.basic.condition.sensitive_catches_dominant
            self.affirmative_action = self.config.basic.condition.affirmative_action
            self.difference_percentage = self.config.basic.condition.difference_percentage
            self.load_from = self.config.data.load_from
            self.csv_to_word = self.config.csv_to_word


        self.df = self._dataloader() if self.load_from is not None else data
        self.sensitive_class_value = None
        self.dominant_class_value = None
        self.class_positive_value = None
        self.class_negative_value = None
        self.label_encoders = {}
        self.features = None
        if self.experiment_name is not  None :
            paths = [self.local_dir_res, self.local_dir_plt]
            for path in paths:
                if not os.path.exists(path):
                    os.makedirs(path)
                else:
                    for root, dirs, files in os.walk(path):
                        for file in files:
                            os.remove(os.path.join(root, file))
                        for directory in dirs:
                            shutil.rmtree(os.path.join(root, directory))

    def _dataloader(self):
        df = pd.read_csv(self.load_from)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        check_data_schema(df,self.class_attribute,self.sensitive_attribute,)
        return df


    def _preprocess(self, df):
        """
            Preprocesses the dataset by encoding categorical features, selecting features,
            splitting the data into training, validation, and test sets, and optionally excluding
            the sensitive attribute.
            """
        self.df, self.label_encoders = encode_dataframe(df)

        self.sensitive_class_value = int(self.label_encoders[self.sensitive_attribute].transform([self.sensitive_attribute_protected])[0])
        self.dominant_class_value = next(value for value in [0, 1] if value != self.sensitive_class_value)
        self.class_positive_value = self.positive_class_value
        self.class_negative_value = [value for value in self.df[self.class_attribute].unique() if value != self.class_positive_value][0]

        if self.feature_selection:
            self.df = backward_regression(df, self.sensitive_attribute)

        self.features = self.df.columns.values.tolist()
        self.features.remove(self.class_attribute)

        train_set, val_set, test_set = split_df(df=self.df,
                                                split_percent=self.split_percent,
                                                protected_attribute=self.sensitive_attribute,
                                                class_attribute=self.class_attribute,
                                                val_data=self.has_val_data,
                                                resampling_train_set=self.resampling_train_set,
                                                sensitive_class_value = self.sensitive_class_value,
                                                dominant_class_value = self.dominant_class_value,
                                                class_positive_value = self.class_positive_value,
                                                class_negative_value = self.class_negative_value,
                                                basic_split= self.basic_split)

        self.x_train, self.y_train, self.y_train_sensitive_attr = get_xy(df=train_set,
                                                        sensitive_attribute=self.sensitive_attribute,
                                                        target_column=self.class_attribute)
        self.x_test, self.y_test, self.y_test_sensitive_attr = get_xy(df=test_set,
                                                       sensitive_attribute=self.sensitive_attribute,
                                                       target_column=self.class_attribute)
        self.x_val, self.y_val, self.y_val_sensitive_attr = get_xy(df=val_set,
                                                    sensitive_attribute=self.sensitive_attribute,
                                                    target_column=self.class_attribute)


        if self.exclude_sensitive_attribute:
            self.x_train = self.x_train.drop(self.sensitive_attribute,axis=1)
            self.x_val = self.x_val.drop(self.sensitive_attribute,axis=1)
            self.x_test = self.x_test.drop(self.sensitive_attribute,axis=1)

        return self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test, self.y_train_sensitive_attr, self.y_val_sensitive_attr, self.y_test_sensitive_attr


    def _train(self):
        """
           Trains a K-Nearest Neighbors (KNN) classifier using the training dataset,
           makes predictions on the validation set, and computes key fairness-related metrics.

           Steps:
           1. Initializes and trains a KNN model using Euclidean distance.
           2. Predicts the class labels for the validation set.
           3. Identifies protected and dominant class samples for negative predictions.
           4. Computes the number of positive predictions for dominant and protected groups.

           Returns:
           None (updates instance attributes).
           """

        self.model = KNeighborsClassifier(n_neighbors=self.knn_neighbors,metric='euclidean')
        self.model.fit(self.x_train, self.y_train[self.class_attribute])

        self.pred_val = self.model.predict(self.x_val)

        self.t0, self.to_ids = get_negative_protected_values(self.pred_val, self.x_val, self.y_val_sensitive_attr,self.class_negative_value, self.sensitive_class_value)
        self.t1, self.t1_ids = get_negative_protected_values(self.pred_val, self.x_val, self.y_val_sensitive_attr,self.class_negative_value, self.dominant_class_value)

        self.sum_positive_pred_dom = len(get_negative_protected_values(self.pred_val, self.x_val, self.y_val_sensitive_attr,self.class_positive_value, self.dominant_class_value)[0])
        self.sum_positive_pred_dom_innit = self.sum_positive_pred_dom
        self.sum_positive_pred_protected = len(get_negative_protected_values(self.pred_val, self.x_val, self.y_val_sensitive_attr,self.class_positive_value, self.sensitive_class_value)[0])
        self.sum_positive_val_dom = len(get_negative_protected_values(self.y_val, self.x_val, self.y_val_sensitive_attr, None,self.dominant_class_value)[0])
        self.sum_positive_val_protected = len(get_negative_protected_values(self.y_val, self.x_val, self.y_val_sensitive_attr, None,self.sensitive_class_value)[0])

    def get_statistics_df(self):
        train_eval = Evaluate(y_actual=self.y_train, y_sensitive_attribute=self.y_train_sensitive_attr,class_attribute=self.class_attribute, sensitive_class_value=self.sensitive_class_value,dominant_class_value=self.dominant_class_value,class_positive_value=self.class_positive_value,class_negative_value=self.class_negative_value,include_pred_stats=False,set_name='train')
        train_statistics = train_eval.get_statistics_dict()

        _, y_total, y_total_sensitive_attr = get_xy(df=self.df,sensitive_attribute=self.sensitive_attribute,target_column=self.class_attribute)
        total_eval = Evaluate(y_actual=y_total, y_sensitive_attribute=y_total_sensitive_attr,class_attribute=self.class_attribute, sensitive_class_value=self.sensitive_class_value,dominant_class_value=self.dominant_class_value,class_positive_value=self.class_positive_value,class_negative_value=self.class_negative_value,include_pred_stats=False,set_name='total')
        total_statistics = total_eval.get_statistics_dict()

        val_eval = Evaluate(y_pred=self.pred_val,y_actual=self.y_val, y_sensitive_attribute=self.y_val_sensitive_attr,class_attribute=self.class_attribute, sensitive_class_value=self.sensitive_class_value,dominant_class_value=self.dominant_class_value,class_positive_value=self.class_positive_value,class_negative_value=self.class_negative_value,set_name='val')
        val_statistics = val_eval.get_statistics_dict()

        stats_df = pd.DataFrame([val_statistics, train_statistics, total_statistics])
        stats_df = stats_df.set_index('name').T
        stats_df = stats_df.where(pd.notnull(stats_df), None)
        stats_df['val_percentage'] = stats_df.apply(lambda row: (row['val'] / row['total'] * 100) if pd.notnull(row['val']) or pd.notnull(row['total']) else None, axis=1)
        stats_df = stats_df[['val', 'val_percentage', 'train', 'total']]
        #stats_df.to_csv(self.experiment_name +'_stats.csv',index=True) and self.experiment_name is not None
        stats_df.to_csv(self.experiment_name.removesuffix("_ida").removesuffix("_eda") +'_stats.csv',index=True) and self.experiment_name is not None
        return stats_df

    def get_test_statistics_df(self,x ,y, y_sensitive_attr):
        y_pred = self.model.predict(x)
        test_eval =Evaluate(y_pred=y_pred,y_actual=y, y_sensitive_attribute=y_sensitive_attr,class_attribute=self.class_attribute, sensitive_class_value=self.sensitive_class_value,dominant_class_value=self.dominant_class_value,class_positive_value=self.class_positive_value,class_negative_value=self.class_negative_value,include_pred_stats=False,set_name='train')
        test_statistics = test_eval.get_test_statistics_df()
        return test_statistics


    def _eval(self,x_data,y_data, y_sensitive_attribute,k=None):
        """
        Evaluates the trained model on a given dataset.

        Parameters:
        x_data (pd.DataFrame): Feature dataset for evaluation.
        y_data (pd.Series or pd.DataFrame): True labels corresponding to x_data.
        y_sensitive_attribute (list): List of sensitive attribute values corresponding to x_data.
        k (int, optional): Iteration count (if applicable for tracking in experiments).

        Returns:
        dict: A dictionary containing the following evaluation results:
            - 'number_sensitive_attr_predicted_positive': Count of positive predictions for the sensitive group.
            - 'pos_t0': Count of positive predictions within the negative protected class group (t0).
            - 'number_dom_attr_predicted_positive': Count of positive predictions for the dominant group.
            - 'number_indices_flipped': Number of flipped predictions in the negative protected class group.
            - 'iteration': Iteration number (if provided).
        """
        y_pred = self.model.predict(x_data)
        eval_results = dict()
        evl = Evaluate(y_actual=y_data,
                        y_pred= y_pred,
                        y_sensitive_attribute=y_sensitive_attribute,
                        class_attribute = self.class_attribute,
                        sensitive_class_value = self.sensitive_class_value,
                        dominant_class_value = self.dominant_class_value,
                        class_positive_value = self.class_positive_value,
                        class_negative_value = self.class_negative_value,
                        )

        eval_results['number_sensitive_attr_predicted_positive'] = evl.number_sensitive_attr_predicted_positive
        eval_results['pos_t0'] = (self.model.predict(self.t0) == self.class_positive_value).sum()
        eval_results['number_dom_attr_predicted_positive'] = evl.number_dom_attr_predicted_positive
        eval_results['number_indices_flipped'] = eval_results['pos_t0']
        eval_results['iteration'] = k


        return eval_results
    def remove_indices_from_train(self, indices):
        """
            Removes specified indices from the training dataset while maintaining
            consistency across features, target labels, and sensitive attributes.

            Parameters:
            indices (int or list of int): Index or list of indices to be removed from the training set.

            Returns:
            None (updates self.x_train, self.y_train, and self.y_train_sensitive_attr).
            """
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
        """
            Identifies and removes nearest negative neighbors of negatively classified
            protected class instances one by one, updating the model iteratively.

            Parameters:
            negative_classified_protected_class (pd.DataFrame, optional):
                Subset of negatively classified protected class samples.
                If None, these are computed dynamically.
            invert_back_to_original (bool, optional):
                If True, restores the training set to its original state after execution.

            Returns:
            list: Indices of negative neighbors removed from the training set.
            """
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
            while True:
                distance, index = self.model.kneighbors(t0.iloc[[i]])
                index_to_be_removed = index[0].tolist()[0]
                vallue = check_column_value(self.y_train, index_to_be_removed, self.class_attribute)
                if vallue == self.class_negative_value:
                    neighbors.append(index_to_be_removed + original_index)
                    self.remove_indices_from_train(index_to_be_removed)
                    self._train()
                else:
                    same_distance_removal = self._get_neighbors_with_same_distance(t0.iloc[[i]],index_to_be_removed)
                    self.remove_indices_from_train(same_distance_removal)
                    t0_negative_neighbors.extend(same_distance_removal)
                    self._train()

                    if invert_back_to_orignal:
                        self.x_train = original_xtrain
                        self.y_train = original_ytrain
                        self.y_train_sensitive_attr = original_y_train_sensitive_attr
                        self._train()

                    t0_negative_neighbors.extend(neighbors)
                    break
        return t0_negative_neighbors

    def _get_neighbors_with_same_distance(self,negative_classified_protected_class,neighbor_index):
        """
            Finds and returns indices of neighbors with the same distance as the specified neighbor,
            but ensures that the neighbors are not classified as positive.

            Parameters:
            negative_classified_protected_class (pd.DataFrame): A DataFrame containing the negative
                                                                 classified protected class sample(s) for which neighbors are to be found.
            neighbor_index (int): Index of the reference neighbor whose distance is to be used as a target.

            Returns:
            list: A list of indices of neighbors with the same distance but not classified as positive.
            """
        distance, index = self.model.kneighbors(negative_classified_protected_class, n_neighbors=350)
        index_position = np.where(index[0] == neighbor_index)[0]
        target_distance = distance[0][index_position]
        same_distance_removal = []
        for i, distance in zip(index[0], distance[0]):
            if i != neighbor_index and distance == target_distance and check_column_value(self.y_train, i,
                                                                                               self.class_attribute) != self.class_positive_value:
                same_distance_removal.append(i)
        return same_distance_removal

    def _get_negative_classified_protected_class_subset(self,df,n_neighbors=384):
        """
            Identifies and returns the subset of neighbors classified as negative for the given
            DataFrame, based on a specified number of nearest neighbors.

            Parameters:
            df (pd.DataFrame): The input DataFrame for which the nearest neighbors are computed.
            n_neighbors (int, optional): The number of nearest neighbors to consider for each sample. Default is 384.

            Returns:
            list: A list of lists, where each sublist contains indices of neighbors classified as negative.
            """
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
                value = check_column_value(self.y_train,element,self.class_attribute)
                if value == self.class_negative_value:
                    new_sublist = sublist[:i+1]
                else:
                    new_sublist = sublist[:i]
                    break

            negative_neighbors.append(new_sublist)
        return negative_neighbors

    def _get_flip_counter(self,train_neighbors):
        """
            Counts the number of neighbors in the training set that are classified as negative.

            Parameters:
            train_neighbors (list): A list of indices representing the neighbors in the training set.

            Returns:
            int: The count of neighbors that are classified as negative.
            """

        counter = 0
        for neighbor in train_neighbors:
            if check_column_value(self.y_train, neighbor, self.class_attribute) == self.class_negative_value:
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
        """
            Filters out the T1 neighbors that are present in the given list of training neighbors.

            Parameters:
            train_neighbors (list): A list of indices representing the neighbors in the training set.

            Returns:
                - dom_attr_indexes: The indices of the filtered T1 neighbors.
                - t1_ids: The corresponding T1 IDs for the filtered neighbors.
            """
        distance, index = self.model.kneighbors(self.t1)

        filtered_t1 = [(lst, self.t1_ids[idx]) for idx, lst in enumerate(index)
                         if any(num in train_neighbors for num in lst)]
        if not filtered_t1:  # This checks if filtered_t1 is an empty list
            return (), ()

        dom_attr_indexes, t1_ids = zip(*filtered_t1)
        return dom_attr_indexes , t1_ids

    def fill_dict_t1(self,reverse_index ,dom_attr_indexes, t1_ids,sensitive_attribute  ):
        """
        Fills the reverse index with Neighbor objects based on the provided T1 data.

        Parameters:
        reverse_index (dict): A dictionary that maps neighbor indices to NeighborContainers.
        dom_attr_indexes (list): A list of lists, where each sublist contains indices of dominant attribute neighbors.
        t1_ids (list): A list of T1 IDs corresponding to the neighbors.
        sensitive_attribute (str): The name of the sensitive attribute.

        Returns:
        dict: The updated reverse index with the new Neighbor objects added.
        """
        for neighbor_id, sublist in enumerate(dom_attr_indexes):
            # Initialize a Neighbor object
            neighbor = Neighbor(
                index=t1_ids[neighbor_id],
                counter_for_flip= self._get_flip_counter(sublist),
                train_neighbors=sublist,
                sensitive_attribute= sensitive_attribute ,
                kneighbors=self.knn_neighbors,
                sensitive_class_value = self.sensitive_class_value,
                dominant_class_value = self.dominant_class_value,
                class_positive_value = self.class_positive_value,
                class_negative_value = self.class_negative_value
            )
            self.val_neighbors.append(neighbor)
            for key in sublist:
                if key in reverse_index:
                    reverse_index[key].add_neighbor(neighbor)
        return reverse_index

    def get_reverse_index(self):
        """
        Retrieves the reverse index for the T0 neighbors by finding their nearest neighbors.

        This method computes the nearest neighbors for the T0 set using the model and initializes
        a reverse index mapping the neighbor indices to their corresponding NeighborContainers.

        Returns:
        dict: A dictionary (reverse_index) mapping neighbor indices to NeighborContainers.
        """
        _, index = self.model.kneighbors(self.t0)
        reverse_index =self.initialize_train_dictionary(index,self.to_ids)
        return  reverse_index



    def lebel_flip_attempt1(self):

        results_df = pd.DataFrame()
        reverse_index = self.get_reverse_index()
        reverse_index = dict(sorted(reverse_index.items(), key=lambda item: len(item[1]), reverse=True))

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
        """
            Initializes a dictionary of training neighbors and adds them to their respective NeighborContainers.

            This method processes a list of neighbor indices and IDs, filters out T1 neighbors,
            and creates `Neighbor` objects for each index. It also updates the reverse index and
            assigns each neighbor to the corresponding `NeighborContainer`.

            Parameters:
            indexes (list): A list of lists, where each sublist contains neighbor indices.
            ids (list): A list of unique IDs corresponding to the neighbor indices.

            Returns:
            dict: A dictionary mapping each neighbor index to its corresponding `NeighborContainer`.
            """
        unique_set = set()
        for lst in indexes:
            unique_set.update(lst)
        train_neighbors = list(unique_set)
        dom_attr_indexes, t1_ids = self.filter_out_t1(train_neighbors)



        dictionary = {}
        for container_key in train_neighbors:
            if check_column_value(self.y_train, container_key, self.class_attribute) == self.class_negative_value:
                container = TrainerKey(container_key,sensitive_class_value=self.sensitive_class_value,
                dominant_class_value=self.dominant_class_value,
                class_positive_value=self.class_positive_value,
                class_negative_value=self.class_negative_value)
                dictionary[container_key] = container

        for neighbor_id, sublist in enumerate(indexes):
            neighbor = Neighbor(
                index=ids[neighbor_id],
                counter_for_flip= self._get_flip_counter(sublist),
                train_neighbors=sublist,
                sensitive_attribute= self.sensitive_class_value ,
                kneighbors=self.knn_neighbors,
                sensitive_class_value=self.sensitive_class_value,
                dominant_class_value=self.dominant_class_value,
                class_positive_value=self.class_positive_value,
                class_negative_value=self.class_negative_value,

            )
            self.val_neighbors.append(neighbor)
            for key in sublist:
                if key in dictionary:
                    dictionary[key].add_neighbor(neighbor)
        dictionary =self.fill_dict_t1(dictionary,dom_attr_indexes, t1_ids,self.dominant_class_value)
        return dictionary

    def make_train_label_positive(self, TrainerKey):
        """
          Adjusts the `counter_for_flip` of neighbors associated with the given TrainerKey.
          If the counter is reduced below zero, a warning is logged.
          Finally, the TrainerKey is removed from the reverse index.

          Parameters:
          TrainerKey (str/int): The key associated with the Trainer in the reverse index.
          """
        for neighbor in self.reverse_index[TrainerKey].neighbors :
            neighbor.counter_for_flip -= 1
            if neighbor.counter_for_flip < 0:
                logging.warning('Counter flip for neighbor {} is negative'.format(neighbor.index))
        self.reverse_index.pop(TrainerKey, None)  # None prevents KeyError if key doesn't exist

    def get_difference_objective(self, pred_sensitive_flipped=None, pred_dom_flipped = None):
        """
            Calculates the difference in positive predictions between the protected and dominant classes.
            The method updates internal counters for positive predictions for both classes
            and computes the difference as a ratio of positive predictions for the protected class
            to those for the dominant class.

            Parameters:
            pred_sensitive_flipped (int, optional): Number of positive predictions flipped for the protected class.
            pred_dom_flipped (int, optional): Number of positive predictions flipped for the dominant class.

            Returns:
            float: The difference objective, calculated as the ratio of positive predictions for the protected class
                   divided by the positive values for the protected class, minus the ratio for the dominant class.
            """
        self.sum_positive_pred_dom += pred_dom_flipped or 0
        self.sum_positive_pred_protected += pred_sensitive_flipped or 0
        self.rpr = self.sum_positive_pred_protected/self.sum_positive_val_protected
        self.bpr = self.sum_positive_pred_dom/self.sum_positive_val_dom

        diff = self.rpr - self.bpr


        return diff

    def _get_current_flips(self):
        """
            Computes the difference in the number of flipped positive predictions between
            the protected and dominant classes compared to the previous state.

            The method updates the total flipped positive counters for both classes and
            calculates the change (delta) in the number of flips for each class since the last call.

            Returns:
                - The difference in flipped positive predictions for the protected class (`curr_protected`).
                - The difference in flipped positive predictions for the dominant class (`curr_dom`).
            """

        prev_protected = self.total_protected_positive_flipped
        prev_dom    = self.total_dom_positive_flipped
        self.total_protected_positive_flipped, self.total_dom_positive_flipped = self.get_flipped_positive_counter()
        curr_protected = self.total_protected_positive_flipped - prev_protected
        curr_dom = self.total_dom_positive_flipped - prev_dom
        return curr_protected, curr_dom



    def label_flip(self):
        """
            This function executes the process of flipping labels in the training dataset tracks the progress of label flips and
            stores the results for each iteration.

            The process involves the following:
            1. Initializing the results DataFrame.
            2. Performing label flips iteratively while the objective condition is met.
            3. Updating the metrics and storing the results.

            Returns:
                - A DataFrame with the results of each iteration (label flips statistics).
                - A list of the objective keys (train indices) used for label flipping.
            """
        results_df = pd.DataFrame()
        train_indexer = []
        self.reverse_index = self.get_reverse_index()
        self.reverse_index_innit = self.reverse_index.copy()
        self.total_protected_positive_flipped, self.total_dom_positive_flipped = self.get_flipped_positive_counter() #should be 0
        diff =self.get_difference_objective()
        if self.experiment_name is not None:

            eval_results = update_results_dict(number_sensitive_attr_predicted_positive =self.sum_positive_pred_protected,
                                               number_sensitive_attr_predicted_negative = len(self.t0),
                                               number_dom_attr_predicted_positive = self.sum_positive_pred_dom,
                                               number_sensitive_attributes_flipped = self.total_protected_positive_flipped,
                                               number_flipped=self.total_protected_positive_flipped + self.total_dom_positive_flipped,
                                               sum_sa_indices_flipped = self.total_protected_positive_flipped,
                                               sum_indices_flipped=0,
                                               rpr = self.rpr,
                                               bpr = self.bpr,
                                               diff = diff,
                                               iteration = len(train_indexer))
            results_df = results_df._append(eval_results, ignore_index=True)

        while self._objective_checker():
            if not self.reverse_index :
                break
            objective_key = self._get_weighted_key()
            train_indexer.append(objective_key)

            self.make_train_label_positive(objective_key)

            curr_protected_flips, curr_dom_flips = self._get_current_flips()
            diff = self.get_difference_objective(curr_protected_flips, curr_dom_flips)
            if self.experiment_name is not None:
                eval_results = update_results_dict(
                    number_sensitive_attr_predicted_positive=self.sum_positive_pred_protected  ,
                    number_sensitive_attr_predicted_negative=len(self.t0)- self.total_protected_positive_flipped ,
                    number_dom_attr_predicted_positive= self.sum_positive_pred_dom ,
                    number_sensitive_attributes_flipped=self.total_protected_positive_flipped,
                    number_flipped = self.total_protected_positive_flipped + self.total_dom_positive_flipped,
                    sum_sa_indices_flipped=curr_protected_flips,
                    sum_indices_flipped = curr_protected_flips + curr_dom_flips,
                    rpr=self.rpr,
                    bpr=self.bpr,
                    diff=diff,
                    iteration=len(train_indexer))

                results_df = results_df._append(eval_results, ignore_index=True)
        results_df.to_csv(self.local_dir_res + 'most_common_flip_results.csv', index=False) and self.experiment_name is not None
        if self.experiment_name is None:
            return  train_indexer
        return results_df , train_indexer

    def _get_weighted_key(self):
        """
            Retrieves the key with the highest weight from the reverse index, based on the specified weight configuration.

            This method iterates over the `reverse_index` to find the item with the maximum weight. The weight is calculated
            depending on whether the configuration includes a dominant attribute or a secondary weight. If multiple items
            have the same weight, the method will return the one with the highest primary weight.

            The method supports the following configurations:
            - If `second_weight` is enabled, the weight is the difference between the primary and secondary weights.
            - If neither is enabled, the weight is just the primary weight.

            Returns:
                result_item (int): The index of the item in the `reverse_index` with the highest calculated weight.
                                   In case of ties, the item with the higher primary weight is selected.
            """
        if self.random_train_point:
            return random.choice(list(self.reverse_index.values())).index
        max_weight = float('-inf')
        keys_with_max_weight = []

        for k, item in self.reverse_index.items():
            item_weight = item.weight  if not self.second_weight else item.weight - item.secondary_weight
            if item_weight  > max_weight:
                max_weight, keys_with_max_weight = item_weight, [item]
            elif item_weight == max_weight:
                keys_with_max_weight.append(item)

        result_item = max(keys_with_max_weight, key=lambda k: item.weight).index

        return result_item


    def _objective_checker(self):
        """
            Checks if the current objective condition has been met based on the affirmative action policy or difference threshold.

            This method evaluates the current state of the model with respect to the predefined objective condition. The condition
            is determined by the configuration's `affirmative_action` setting and a difference threshold percentage.
            The method returns `True` if the objective condition is satisfied, otherwise it returns `False`.

            The check is based on two conditions:
            1. If `affirmative_action` is enabled:
               - The method calculates the allowed difference as a percentage of the initial number of positive predictions for the dominant class.
               - It returns `True` if the difference between the initial dominant class predictions and the protected class predictions is greater than or equal to the allowed difference.
            2. If `affirmative_action` is not enabled:
               - The method calculates the difference objective (using `get_difference_objective()`), and compares it to the threshold (adjusted by a small value of 0.005).
               - It returns `True` if the difference is smaller than the threshold, and `False` otherwise.

            Returns:
                bool: `True` if the objective condition is satisfied, `False` otherwise.
            """
        if self.sensitive_catches_dominant:
            return self.sum_positive_pred_dom - self.sum_positive_pred_protected > 0
        if self.affirmative_action:
            allowed_difference = (self.sum_positive_pred_dom_innit * self.difference_percentage) // 100
            return self.sum_positive_pred_dom_innit - self.sum_positive_pred_protected >= allowed_difference
        else:
            threshold = self.difference_percentage / 100
            diff = self.get_difference_objective()
            return diff <= threshold

    def get_flipped_positive_counter(self):
        """
            Counts the number of positive-classified instances that have been flipped for both the protected and dominant class attributes.

            This method iterates through the list of validation neighbors (`self.val_neighbors`) and counts how many instances
            have been predicted as the positive class (`self.class_positive_value`). The counts are divided into two categories:
            1. `total_protected_positive`: The number of flipped positive instances belonging to the sensitive (protected) class.
            2. `total_dom_positive`: The number of flipped positive instances belonging to the dominant class.

            Returns:
                    - `total_protected_positive` (int): The number of flipped positive instances for the protected class.
                    - `total_dom_positive` (int): The number of flipped positive instances for the dominant class.
            """
        total_protected_positive = 0
        total_dom_positive = 0
        for neighbor in self.val_neighbors:
            if neighbor.predicted_label == self.class_positive_value:
                if neighbor.sensitive_attribute == self.sensitive_class_value:
                    total_protected_positive += 1
                elif neighbor.sensitive_attribute == self.dominant_class_value:
                    total_dom_positive += 1
        return total_protected_positive, total_dom_positive

    def run_fairness_parity(self):

        self._preprocess(self.df)
        self._train()
        self.get_statistics_df()
        ####
        test_stas_before = self.get_test_statistics_df(self.x_val, self.y_val, self.y_val_sensitive_attr)
        ####
        reslts_df , train_indexer =self.label_flip()

        rename_columns_(reslts_df,self.local_dir_res + self.experiment_name+'_'+'most_common_flip_results.csv') and self.csv_to_word
        if self.experiment_name is None:
            self.y_train = flip_value(self.y_train, train_indexer, self.class_positive_value, self.config.data.class_attribute.name)
            self._train()  # Retrain
            return train_indexer if self.return_model is False else self.model

        #Stats####
        '''
        self.y_train = flip_value(self.y_train, train_indexer, self.class_positive_value,
                                  self.class_attribute)
        self._train()  # Retrain
        '''
        test_stas_after = self.get_test_statistics_df(self.x_val, self.y_val, self.y_val_sensitive_attr)
        test_stats = export_test_statistics(test_stas_before, test_stas_after)
        ######
        return reslts_df, train_indexer ,test_stats
