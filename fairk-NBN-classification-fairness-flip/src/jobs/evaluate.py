import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


class Evaluate():

    def __init__(self, y_pred=None, y_actual=None,y_sensitive_attribute=None,class_attribute= None):
        self.y_pred = y_pred
        self.y_actual = y_actual[class_attribute].tolist()
        self.y_sensitive_attribute = y_sensitive_attribute
        self.confusion_matrix = confusion_matrix(self.y_actual,self.y_pred)
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

        for i in range(len(self.y_pred)):
            if self.y_actual[i] == self.y_pred[i] == 1:
                self.tp += 1
            if self.y_pred[i] == 1 and self.y_actual[i] != self.y_pred[i]:
                self.fp += 1
            if self.y_actual[i] == self.y_pred[i] == 2:
                self.tn += 1
            if self.y_pred[i] == 2 and self.y_actual[i] != self.y_pred[i]:
                self.fn += 1

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
        y_pred_z0 = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == 0 ]
        y_pred_z1 = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == 1 ]
        y_pred_z0_positive_count = y_pred_z0.count(1)
        y_pred_z1_positive_count = y_pred_z1.count(1)
        fairness_parity_rate = y_pred_z0_positive_count / y_pred_z1_positive_count
        return fairness_parity_rate
    def true_negative_fairness_rate(self):
        y_pred_z0 = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == 0 and self.y_pred[i] == self.y_actual[i] ]
        y_pred_z1 = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if self.y_sensitive_attribute[i] == 1 and self.y_pred[i] == self.y_actual[i]]
        y_pred_z0_negative_count = y_pred_z0.count(2)
        y_pred_z1_negative_count = y_pred_z1.count(2)
        if y_pred_z0_negative_count == 0 or y_pred_z1_negative_count == 0:
            true_negative_fairness_rate = 0
        else:
            true_negative_fairness_rate = y_pred_z0_negative_count / y_pred_z1_negative_count
        return true_negative_fairness_rate
    def true_positive_fairness_rate(self):
        y_pred_z0 = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if
                     self.y_sensitive_attribute[i] == 0 and self.y_pred[i] == self.y_actual[i]]
        y_pred_z1 = [self.y_pred[i] for i in range(len(self.y_sensitive_attribute)) if
                     self.y_sensitive_attribute[i] == 1 and self.y_pred[i] == self.y_actual[i]]
        y_pred_z0_positive_count = y_pred_z0.count(1)
        y_pred_z1_positive_count = y_pred_z1.count(1)
        true_positive_fairness_rate = y_pred_z0_positive_count / y_pred_z1_positive_count
        return true_positive_fairness_rate


    def get_accuracy(self):
        accuracy = accuracy_score(y_true=self.y_actual, y_pred=self.y_pred)
        return accuracy