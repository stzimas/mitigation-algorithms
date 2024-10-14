import logging
import os
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import omegaconf
import sys
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tukeyplot import results

from src.dataloader.dataloader import DataLoader
from src.config.schema import Config
from src.jobs.evaluate import Evaluate
from src.utils.general_utils import get_neighbor_statistics, get_negative_protected_values, check_column_value, \
    group_sublists_by_shared_elements
from src.utils.preprocess_utils import encode_dataframe, split_df, get_xy, backward_regression


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
        #self.scaler = StandardScaler()
        #self.scaler.fit(train_set[self.features])

        #train_set[self.features] = self.scaler.transform(train_set[self.features])
        #test_set[self.features] = self.scaler.transform(test_set[self.features])
        x_train, y_train, train_sensitive_attr = get_xy(df=train_set,
                                                        sensitive_attribute=self.config.data.sensitive_attribute.name,
                                                        target_column=self.config.data.class_attribute.name)
        x_test, y_test, y_test_sensitive_attr = get_xy(df=test_set,
                                                       sensitive_attribute=self.config.data.sensitive_attribute.name,
                                                       target_column=self.config.data.class_attribute.name)
        x_val, y_val, y_val_sensitive_attr = get_xy(df=val_set,
                                                    sensitive_attribute=self.config.data.sensitive_attribute.name,
                                                    target_column=self.config.data.class_attribute.name)
        if self.config.basic.exclude_sensitive_attribute:
            x_train = x_train.drop(self.sensitive_attribute,axis=1)
            x_val = x_val.drop(self.sensitive_attribute,axis=1)
            x_test = x_test.drop(self.sensitive_attribute,axis=1)

        return x_train, x_val, x_test, y_train, y_val, y_test, train_sensitive_attr, y_val_sensitive_attr, y_test_sensitive_attr


    def _train(self, x_train, y_train, sns=None):
        cov_matrix = np.cov(x_train.T)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        self.model = KNeighborsClassifier(n_neighbors=self.config.basic.neighbors,metric='l2')
        self.model.fit(x_train, y_train[self.class_attribute])

    def _eval(self, x_test, y_test, y_sensitive_attr):
        y_pred = self.model.predict(x_test)
        eval_results = dict()

        eval = Evaluate(y_actual=y_test, y_pred=y_pred, y_sensitive_attribute=y_sensitive_attr,
                        class_attribute=self.class_attribute)
        eval_results['fairness_parity_rate'] = eval.fairness_parity_rate()
        eval_results['accuracy'] = eval.get_accuracy()
        #eval_results['positive_predictive_value'] = eval.get_positive_predictive_value()
        eval_results['true_positive_fairness_rate'] = eval.true_positive_fairness_rate()
        eval_results['negative_fairness_representation_rate'] = eval.negative_fairness_representation_rate()
        eval_results['negative_sensitive_val_counter'] = eval.negative_sensitive_class_counter
        eval_results['accuracy_fair_rate'] = eval.acc_fair_rate

        return eval_results


    def _get_negative_classified_protected_class_subset(self,df,y_train,n_neighbors=384):
        distances, indices = self.model.kneighbors(df, n_neighbors=n_neighbors,return_distance=True)
        negative_neighbors = []
        for sublist in indices:
            new_sublist = []
            for i,element in enumerate(sublist):
                value = check_column_value(y_train,element,self.config.data.class_attribute.name)
                if value == 2:
                    new_sublist = sublist[:i+1]

                else:
                    # If no condition was met, keep the full sublist
                    new_sublist = sublist[:i]
                    break

            negative_neighbors.append(new_sublist)

        return negative_neighbors

    def attempt0(self, x_train, x_test, y_train, y_test, y_sensitive_attr):
        results_df = pd.DataFrame()
        iteration = 0
        eval_results = {self.config.basic.focus_metric: 0}
        while True:  #eval_results[self.config.basic.focus_metric] <= 1.0
            iteration += 100
            self._train(x_train, y_train)
            eval_results = self._eval(x_test, y_test, y_sensitive_attr)
            eval_results['Iteration'] = iteration
            print(eval_results)
            results_df = results_df._append(eval_results, ignore_index=True)
            indices = x_train[(x_train['Sex'] == 0) & (y_train['class'] == 2)].index
            print(len(indices))
            if len(indices) == 0:
                break
            random_index = np.random.choice(indices, size=100)
            x_train = x_train.drop(random_index)
            y_train = y_train.drop(random_index)
            if indices.empty:  #eval_results.get(self.config.basic.focus_metric) >= 1.0 - 1e-10
                break
        results_df.to_csv(self.local_dir_res + 'eval_results_coverage.csv', index=False)
        return results_df

    def coverage_visualization(self, results_df, metric='fairness_parity_rate'):
        results_df = results_df.drop(columns=['negative_sensitive_val_counter','points_dropped_from_train','accuracy_fair_rate'], axis=1)
        fig, ax = plt.subplots()
        for column in results_df.columns:
            if column not in ['Iteration']:
                if column is not metric:
                    ax.plot(results_df['Iteration'], results_df[column], label=column,
                            linewidth=1.3)
                else:
                    ax.plot(results_df['Iteration'], results_df[column], label=column,
                            linewidth=2)
        try:
            first_fairness_parity_rate_index = results_df[results_df[metric] >= 1.0 - 1e-10].index[0]
            ax.axvline(x=results_df.loc[first_fairness_parity_rate_index, 'Iteration'], color='r', linestyle='--',
                       label='Fairness Parity Rate = 1.0')
        except IndexError:
            print("Not 1.0 value found in the results")

        y_min, y_max = results_df.drop(columns='Iteration').values.min(), results_df.drop(
            columns='Iteration').values.max()
        y_ticks = np.linspace(y_min, y_max,
                              20)
        ax.set_yticks(y_ticks)
        x_min, x_max = results_df['Iteration'].min(), results_df['Iteration'].max()
        #x_ticks = np.arange(x_min, x_max + 1, 5)  # Ticks from min to max, with step of 5
        #ax.set_xticks(x_ticks)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Evaluation Results')
        ax.legend(fontsize=12)
        ax.set_title('Evaluation Metrics Over Iterations')
        plt.tight_layout(pad=0.5)
        plt.savefig(self.local_dir_plt + 'evaluation_metrics_plot.png', dpi=1200, bbox_inches='tight', pad_inches=0.04)

    def attempt1(self, x_train, x_val, x_test, y_train, y_val, y_test, train_sensitive_attr,
                                      y_val_sensitive_attr, y_test_sensitive_attr):
        results_df = pd.DataFrame()
        iteration = 0
        eval_results = {self.config.basic.focus_metric: 0}
        while True:
            iteration += 1
            t0 = get_negative_protected_values(y_val, x_val, y_val_sensitive_attr, self.class_attribute)
            indices = self._get_neighbors(t0)

            neighbor_protected_class_counter = sum(1 for idx in indices if train_sensitive_attr[idx] == 0)
            neighbor_protected_class_counter_percentage = neighbor_protected_class_counter / len(t0)

            x_train = x_train.drop(indices)
            x_train.reset_index(drop=True, inplace=True)
            train_sensitive_attr = [item for idx, item in enumerate(train_sensitive_attr) if idx not in indices]
            y_train = y_train.drop(indices)
            y_train.reset_index(drop=True, inplace=True)
            if (len(x_train) == 1):
                break
            eval_results = self._eval(x_val, y_val, y_val_sensitive_attr)

            eval_results['Iteration'] = iteration
            eval_results['neighbor_protected_class_counter_percentage'] = neighbor_protected_class_counter_percentage
            eval_results['points_dropped_from_train'] = len(Counter(indices))
            logging.info(eval_results)

            results_df = results_df._append(eval_results, ignore_index=True)
            if eval_results.get(self.config.basic.focus_metric) >= 1.0 :
                break
            #retrain
            self._train(x_train, y_train)
        results_df = results_df.round(4)
        test_results = self._eval(x_test, y_test, y_test_sensitive_attr)
        logging.info("test results:")
        logging.info(test_results)
        logging.info("total points removed from train set:"+str(results_df['points_dropped_from_train'].sum()))
        results_df.to_csv(self.local_dir_res + 'eval_results_coverage.csv', index=False)

        return results_df

    def most_common_flip(self, y_val=None, x_val=None, y_val_sensitive_attr=None, y_train=None, x_train=None,
                         number_sensitive_attr_predicted_positive=None, number_dom_attr_predicted_positive=None,
                         y_test=None):
        num_flips = 0
        results_df = pd.DataFrame()
        number_sensitive_attr_predicted_positive_prev_interation = number_sensitive_attr_predicted_positive
        max_length = self.model.n_samples_fit_
        while number_sensitive_attr_predicted_positive != number_dom_attr_predicted_positive:
            results_dict = dict()
            t0 = get_negative_protected_values(y_val, x_val, y_val_sensitive_attr, self.class_attribute)
            negative_clasified_protected_class_subset = self._get_negative_classified_protected_class_subset(t0, y_train,n_neighbors =max_length)

            max_length = max(len(sublist) for sublist in negative_clasified_protected_class_subset)
            if max_length < 15:
                print("hi")
                test = self._get_negative_classified_protected_class_subset(t0, y_train, n_neighbors=self.model.n_samples_fit_)
                pass
            print(f"max_length: {max_length}")
            index_0_elements =  [sublist[0] for sublist in negative_clasified_protected_class_subset if (len(sublist)>0)]
            counter = Counter(index_0_elements)
            most_common_index, count = counter.most_common(1)[0]
            results_dict['Index_Flipped'] = most_common_index
            results_dict['num_pneg_neighbors'] = count
            print(f"most_common_index: {most_common_index}")
            print(f"count: {count}")
            y_train.loc[most_common_index,self.class_attribute] = 1 #flip label
            self._train(x_train, y_train)
            y_pred_val = self.model.predict(x_val)
            print(f"positive_t0: {(self.model.predict(t0) == 1).sum()}")
            eval = Evaluate(y_actual=y_val, y_pred=y_pred_val, y_sensitive_attribute=y_val_sensitive_attr,
                            class_attribute=self.class_attribute)

            number_sensitive_attr_predicted_positive = eval.number_sensitive_attr_predicted_positive
            number_pneg_positive = eval.number_sensitive_attr_predicted_positive - number_sensitive_attr_predicted_positive_prev_interation
            number_sensitive_attr_predicted_positive_prev_interation = eval.number_sensitive_attr_predicted_positive
            results_dict['pnegs_flipped'] = number_pneg_positive

            number_dom_attr_predicted_positive = eval.number_dom_attr_predicted_positive
            results_dict['number_sensitive_attr_predicted_positive'] = number_sensitive_attr_predicted_positive
            results_dict['number_dom_attr_predicted_positive'] = number_dom_attr_predicted_positive
            num_flips += 1
            results_dict['num_flips'] = num_flips
            results_df = results_df._append(results_dict, ignore_index=True)

            number_sensitive_attr_predicted_negative = eval.number_sensitive_attr_predicted_negative
            number_dom_attr_predicted_negative = eval.number_dom_attr_predicted_negative

            print(f"number_sensitive_attr_predicted_positive: {number_sensitive_attr_predicted_positive}")
            print(f"number_sensitive_attr_predicted_negative: {number_sensitive_attr_predicted_negative}")
            print(f"number_dom_attr_predicted_positive: {number_dom_attr_predicted_positive}")
            print(f"number_dom_attr_predicted_negative: {number_dom_attr_predicted_negative}")
            if all(len(sublist) == 0 for sublist in negative_clasified_protected_class_subset):
                break

        results_df.to_csv(self.local_dir_res + 'most_common_flip_results.csv', index=False)
        return results_df

    def run_fairness_par(self):
        dataloader = DataLoader()
        df = dataloader.load_data(config=self.config)

        x_train, x_val, x_test, y_train, y_val, y_test, x_train_sensitive_attr, y_val_sensitive_attr, y_test_sensitive_attr = self._preprocess(
            df)
        self._train(x_train, y_train)
        y_pred_val = self.model.predict(x_val)

        eval = Evaluate(y_actual=y_val, y_pred=y_pred_val, y_sensitive_attribute=y_val_sensitive_attr,
                        class_attribute=self.class_attribute)
        number_sensitive_attr_predicted_positive = eval.number_sensitive_attr_predicted_positive
        number_dom_attr_predicted_positive = eval.number_dom_attr_predicted_positive
        results = self.most_common_flip( y_val, x_val, y_val_sensitive_attr, y_train, x_train,
                         number_sensitive_attr_predicted_positive, number_dom_attr_predicted_positive,
                         y_test)



        #self.coverage_visualization(results_df, metric=self.config.basic.focus_metric)
