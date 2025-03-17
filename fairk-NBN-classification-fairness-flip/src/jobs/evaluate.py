import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score



class Evaluate():

    def  __init__(self, y_pred=None,
                  y_actual=None,
                  y_sensitive_attribute=None,
                  class_attribute= None,
                  sensitive_class_value = None ,
                  dominant_class_value = None ,
                  class_positive_value = None ,
                  class_negative_value = None,
                  include_pred_stats = True,
                  set_name = None

    ):
        self.set_name = set_name
        self.include_pred_stats = include_pred_stats
        self.y_pred = np.ones_like(y_actual) if y_pred is None else y_pred
        self.y_actual = y_actual[class_attribute].tolist()
        self.y_sensitive_attribute = y_sensitive_attribute
        self.confusion_matrix = confusion_matrix(self.y_actual,self.y_pred)
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.negative_sensitive_class_counter = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if
                     self.y_sensitive_attribute[i] == sensitive_class_value].count(class_negative_value)

        self.sum_sensitive_attr = len([self.y_sensitive_attribute[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == sensitive_class_value])
        self.sum_dom_attr = len([self.y_sensitive_attribute[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == dominant_class_value])
        self.sum_positive = len([self.y_actual[i] for i in range(len(self.y_actual)) if self.y_actual[i] == class_positive_value])
        self.sum_negative = len([self.y_actual[i] for i in range(len(self.y_actual)) if self.y_actual[i] == class_negative_value])
        self.sum_negative_sensitive_attr = [self.y_actual[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == sensitive_class_value ].count(class_negative_value)
        self.sum_positive_sensitive_attr = [self.y_actual[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == sensitive_class_value ].count(class_positive_value)
        self.sum_negative_dom_attr = [self.y_actual[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == dominant_class_value].count(class_negative_value)
        self.sum_positive_dom_attr = [self.y_actual[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == dominant_class_value].count(class_positive_value)

        self.sum_pred_positive = len([self.y_pred[i] for i in range(len(self.y_pred)) if self.y_pred[i] == class_positive_value])
        self.sum_pred_negative = len([self.y_pred[i] for i in range(len(self.y_pred)) if self.y_pred[i] == class_negative_value])



        self.tp_sensitive_attr = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == sensitive_class_value and self.y_pred[i] == self.y_actual[i]].count(class_positive_value)
        self.fp_sensitive_attr = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == sensitive_class_value and self.y_pred[i] != self.y_actual[i]].count(class_positive_value)
        self.tn_sensitive_attr = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == sensitive_class_value and self.y_pred[i] == self.y_actual[i]].count(class_negative_value)
        self.fn_sensitive_attr = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == sensitive_class_value and self.y_pred[i] != self.y_actual[i]].count(class_negative_value)

        self.tp_dom_attr = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == dominant_class_value and self.y_pred[i] ==self.y_actual[i]].count(class_positive_value)
        self.fp_dom_attr = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == dominant_class_value and self.y_pred[i] != self.y_actual[i]].count(class_positive_value)
        self.tn_dom_attr = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == dominant_class_value and self.y_pred[i] == self.y_actual[i]].count(class_negative_value)
        self.fn_dom_attr = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == dominant_class_value and self.y_pred[i] != self.y_actual[i]].count(class_negative_value)


        self.sen_attr_ppv =  self.tp_sensitive_attr / (self.tp_sensitive_attr + self.fp_sensitive_attr)
        self.dom_attr_ppv =  self.tp_dom_attr / (self.tp_dom_attr + self.fp_dom_attr)


        self.number_sensitive_attr_predicted_positive = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == sensitive_class_value ].count(class_positive_value)
        self.number_sensitive_attr_predicted_negative = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == sensitive_class_value ].count(class_negative_value)


        self.number_dom_attr_predicted_positive = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == dominant_class_value ].count(class_positive_value)
        self.number_dom_attr_predicted_negative = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == dominant_class_value ].count(class_negative_value)

        self.sen_attr_positive_ratio = self.number_sensitive_attr_predicted_positive / (self.number_sensitive_attr_predicted_positive+self.number_sensitive_attr_predicted_negative)
        self.dom_attr_positive_ratio = self.number_dom_attr_predicted_positive / (self.number_dom_attr_predicted_positive+self.number_dom_attr_predicted_negative)
        self.ratio_diff = self.sen_attr_positive_ratio - self.dom_attr_positive_ratio


        for i in range(len(self.y_pred)):
            if self.y_actual[i] == self.y_pred[i] == 1:
                self.tp += 1
            if self.y_pred[i] == 1 and self.y_actual[i] != self.y_pred[i]:
                self.fp += 1
            if self.y_actual[i] == self.y_pred[i] == 2:
                self.tn += 1
            if self.y_pred[i] == 2 and self.y_actual[i] != self.y_pred[i]:
                self.fn += 1

    def get_statistics_dict(self):
        statistics = {
            'sum_red': self.sum_sensitive_attr,
            'sum_blue': self.sum_dom_attr,
            'sum_positive': self.sum_positive,
            'sum_negative': self.sum_negative,
            'sum_negative_red': self.sum_negative_sensitive_attr,
            'sum_positive_red': self.sum_positive_sensitive_attr,
            'sum_negative_blue': self.sum_negative_dom_attr,
            'sum_positive_blue': self.sum_positive_dom_attr,
        }

        if self.include_pred_stats:
            statistics.update({
                'sum_pred_positive': self.sum_pred_positive,
                'sum_pred_negative': self.sum_pred_negative,
                'sum_red_predicted_positive': self.number_sensitive_attr_predicted_positive,
                'sum_red_predicted_negative': self.number_sensitive_attr_predicted_negative,
                'sum_blue_predicted_positive': self.number_dom_attr_predicted_positive,
                'sum_blue_predicted_negative': self.number_dom_attr_predicted_negative,
                'red_ppv': self.sen_attr_ppv,
                'blue_ppv': self.dom_attr_ppv,
            })
        if self.set_name is not None:
            statistics.update({'name': self.set_name})

        return statistics
    def get_test_statistics_df(self):
        statistics = {
            'sen_attr_positive_ratio': self.sen_attr_positive_ratio,
            'dom_attr_positive_ratio': self.dom_attr_positive_ratio,
            'difference': self.dom_attr_positive_ratio - self.sen_attr_positive_ratio,
            'sen_attr_ppv': self.sen_attr_ppv,
            'dom_attr_ppv': self.dom_attr_ppv,
            'accuracy': self.get_accuracy()
        }
        return statistics


    def get_true_positve(self, confusion_matrix):
        tp = confusion_matrix[1, 1]
        return tp

    def get_false_positive(self, confusion_matrix):
        fp = confusion_matrix[0, 1]
        return fp

    def get_false_negative(self, confusion_matrix):
        fn = confusion_matrix[1, 0]
        return fn

    def get_true_negative(self, confusion_matrix, fp, fn, tp):
        tn = confusion_matrix[0, 0]
        return tn

    def get_positive_predictive_value(self):

        ppv = self.tp / (self.tp + self.fp)
        return ppv

    def get_false_discovery_rate(self):
        fdr = self.fp / (self.tp + self.fp)
        return fdr

    def false_omission_rate(self):
        f_om_r = self.fn / (self.tn + self.fn)
        return f_om_r

    def negative_predictive_value(self):
        npv = self.tn / (self.tn + self.fn)
        return npv

    def true_positive_rate(self):
        tpr = self.tp / (self.tp + self.fn)
        return tpr

    def false_positive_rate(self):
        fpr = self.fp / (self.fp + self.tn)
        return fpr

    def false_negative_rate(self):
        fnr = self.fn / (self.tp + self.fn)
        return fnr

    def true_negative_rate(self):
        tnr = self.tn / (self.fp + self.tn)
        return tnr

    def fairness_parity_rate(self):
        y_pred_z0 = self.tp_sensitive_attr + self.fp_sensitive_attr
        y_pred_z1 = self.tp - self.tp_sensitive_attr  + self.fp - self.fp_sensitive_attr
        fairness_parity_rate = y_pred_z0 / y_pred_z1
        return fairness_parity_rate
    def negative_fairness_representation_rate(self):
        y_pred_z0 = self.fn_sensitive_attr
        y_pred_z1 = self.fn - self.fn_sensitive_attr

        if self.fn - self.fn_sensitive_attr == 0:
            negative_fairness_representation_rate = 0
        else:
            negative_fairness_representation_rate = y_pred_z0 / y_pred_z1
        return negative_fairness_representation_rate
    def true_positive_fairness_rate(self):
        y_pred_z0 = self.tp_sensitive_attr
        y_pred_z1 = self.tp -self.tp_sensitive_attr
        true_positive_fairness_rate = y_pred_z0 / y_pred_z1
        return true_positive_fairness_rate


    def get_accuracy(self):
        accuracy = accuracy_score(y_true=self.y_actual, y_pred=self.y_pred)
        return accuracy