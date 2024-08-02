import os
import shutil

import numpy as np
import pandas as pd
import omegaconf
import sys
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from src.dataloader.dataloader import DataLoader
from src.config.schema import Config
from src.jobs.evaluate import Evaluate
from src.utils.preprocess_utils import encode_dataframe, split_df, get_xy, backward_regression


class FairnessParity:
    def __init__(self, config: Config):
        self.config = config
        self.experiment_name = self.config.experiment_name
        self.local_dir_res = 'data/'+self.experiment_name +'/' +self.config.data.results_path
        self.local_dir_plt = 'data/'+self.experiment_name +'/' +self.config.data.plot_path
        self.sensitive_attribute = self.config.data.sensitive_attribute.name
        self.class_attribute = self.config.data.class_attribute.name
        self.categorical_attribute = None
        self.label_encoders = {}
        self.features = None
        paths = [self.local_dir_res,self.local_dir_plt]
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
            df = backward_regression(df,self.sensitive_attribute)

        self.features = df.columns.values.tolist()
        self.features.remove(self.class_attribute)

        train_set, test_set = split_df(df=df,
                                       split_percent=self.config.basic.split_percent,
                                       protected_attribute=self.config.data.sensitive_attribute.name,
                                       val_data=False,
                                       resampling_train_set=self.config.basic.split_data['resampling_train_set'],)
        #self.scaler = StandardScaler()
        #self.scaler.fit(train_set[self.features])

        #train_set[self.features] = self.scaler.transform(train_set[self.features])
        #test_set[self.features] = self.scaler.transform(test_set[self.features])
        x_train, y_train, _ = get_xy(df=train_set,sensitive_attribute=self.config.data.sensitive_attribute.name, target_column=self.config.data.class_attribute.name)
        x_test, y_test, y_sensitive_attr = get_xy(df=test_set,sensitive_attribute=self.config.data.sensitive_attribute.name, target_column=self.config.data.class_attribute.name)

        return x_train, x_test, y_train, y_test , y_sensitive_attr

    def _train(self, x_train, y_train, sns=None):
        self.model = KNeighborsClassifier(n_neighbors=self.config.basic.neighbors)
        self.model.fit(x_train, y_train[self.class_attribute])




    def _eval(self,x_test,y_test,y_sensitive_attr):
        y_pred = self.model.predict(x_test)
        eval_results = dict()

        eval = Evaluate(y_actual=y_test, y_pred=y_pred,y_sensitive_attribute=y_sensitive_attr, class_attribute=self.class_attribute)
        eval_results['fairness_parity_rate'] = eval.fairness_parity_rate()
        eval_results['accuracy'] = eval.get_accuracy()
        eval_results['positive_predictive_value'] = eval.get_positive_predictive_value()
        eval_results['true_positive_fairness_rate'] = eval.true_positive_fairness_rate()
        eval_results['true_negative_fairness_rate'] = eval.true_negative_fairness_rate()

        return eval_results

    def _run_fairness_parity_coverage(self,x_train, x_test, y_train, y_test, y_sensitive_attr):
        results_df = pd.DataFrame()
        iteration = 0
        eval_results = {self.config.basic.focus_metric: 0}
        while True: #eval_results[self.config.basic.focus_metric] <= 1.0
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
            random_index = np.random.choice(indices,size=100)
            x_train = x_train.drop(random_index)
            y_train = y_train.drop(random_index)
            if  indices.empty: #eval_results.get(self.config.basic.focus_metric) >= 1.0 - 1e-10
                break
        results_df.to_csv(self.local_dir_res+'eval_results_coverage.csv', index=False)
        return results_df


    def coverage_visualization(self, results_df, metric = 'fairness_parity_rate'):
        fig, ax = plt.subplots()
        for column in results_df.columns:
            if column != 'Iteration':
                if column is not metric:
                    ax.plot(results_df['Iteration'], results_df[column], label=column,
                            linewidth=1.3)
                else:
                    ax.plot(results_df['Iteration'], results_df[column], label=column,
                            linewidth=2)
        try:
            first_fairness_parity_rate_index = results_df[results_df[metric] >= 1.0 - 1e-10].index[0]
            ax.axvline(x=results_df.loc[first_fairness_parity_rate_index, 'Iteration'], color='r', linestyle='--',label='Fairness Parity Rate = 1.0')
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
        ax.set_xlabel('Negative Values of the Sensitive Attribute removed')
        ax.set_ylabel('Evaluation Results')
        ax.legend(fontsize=12)
        ax.set_title('Evaluation Metrics Over Iterations')
        plt.tight_layout(pad=0.5)
        plt.savefig(self.local_dir_plt + 'evaluation_metrics_plot.png', dpi=1200, bbox_inches='tight', pad_inches=0.1)

    def run_fairness_par(self):
        dataloader = DataLoader()
        df = dataloader.load_data(config=self.config)

        x_train, x_test, y_train, y_test, y_sensitive_attr = self._preprocess(df)
        self._train(x_train,y_train)
        results = self._eval(x_test,y_test,y_sensitive_attr)
        results_df = self._run_fairness_parity_coverage(x_train, x_test, y_train, y_test, y_sensitive_attr)
        self.coverage_visualization(results_df, metric=self.config.basic.focus_metric)


