import argparse
import os.path
import statistics

import numpy as np
import pandas as pd

from util import *

from typing import Sequence

from sting.classifier import Classifier
from sting.data import AbstractDataSet, parse_c45, Feature

np.random.seed(12345)


class LogReg(Classifier):
    # constructor for logreg functio9n
    def __init__(self, dataset, lam):
        self.weights = np.random.rand(len(dataset.columns))
        self.data = dataset
        self.lam = lam
        self.nom_bins = None

    # trains the functions
    def train(self) -> None:
        self.data = change_nominals(self.data, self)
        self.data['ones'] = [-1 for i in range(0, len(self.data))]
        for i in range(0, len(self.data)):
            self.weights = update_weights(self.weights, self.data, self.lam)

    # predicts the data using logreg
    def predict(self, data) -> Sequence[int]:
        get_values(data, self)
        data['ones'] = [-1 for i in range(0, len(data))]
        conf = h(self.weights, data)
        prediction = greater_than_chance(conf)
        return prediction, conf


# assigns predicted data into correct bins
def get_values(data, classifier):
    data.reset_index(inplace=True, drop=True)
    for col in data.select_dtypes(exclude='float').columns.values:
        new = np.array([])
        for i in range(0, len(data)):
            val = data.loc[i, col]
            try:
                number = classifier.nom_bins[col][val]
            except KeyError:
                number = max(classifier.num_bins[col]) + 1
                classifier.num_bins[col][val] = number
            new = np.append(new,number)
        data[col] = new


# updates the weights
def update_weights(weights, data, lamba):
    pos = data[data.label == 1]
    pos.reset_index(inplace=True, drop=True)
    pos = pos.drop("label", axis=1)
    data = data.drop("label", axis=1)
    dL = lamba * weights + np.array(h(weights, data).transpose() @ data) - np.array([sum(pos[x]) for x in pos.columns])
    return weights - (0.1 * dL)


# sigmoid activation function
def h(weights, features):
    return (1 + np.e ** -(features @ weights)) ** -1

# returns bernoulli variable for one half
def greater_than_chance(arr):
    arr = np.array(arr)
    for i in range(0, len(arr)):
        if arr[i] >= 0.5:
            arr[i] = 1
        else:
            arr[i] = 0
    return arr

# puts nominals into classes
def change_nominals(data, classifier):
    df = data.select_dtypes(exclude=float)
    classifier.nom_bins = {x: None for x in df.columns.values}
    for feature in df.columns:
        classifier.nom_bins[feature] = {x: None for x in np.unique(df[feature])}
        data[feature] = update_nominal(data[feature], classifier, feature)
    return data


# updates a vector of the nominals
def update_nominal(nominal, classifier, name):
    nominal = np.array(nominal)
    values = np.unique(nominal)
    i = 1
    output = np.array([1 for x in nominal])
    for value in values:
        classifier.nom_bins[name][value] = i
        output[nominal == value] = i
        i += 1
    return output


def standard_deviation(arr: [float]):
    return statistics.stdev(arr)


def average_value(arr: [float]):
    return sum(arr) / len(arr)


# prints out the metrix for logreg function
def evaluate_logreg(Accuracy: [float], Precision: [float], Recall: [float], area_under_curve: float):
        if len(Accuracy) == 1:
            stdevA = 0
            stdevP = 0
            stdevR = 0
        else:
            stdevA = standard_deviation(Accuracy)
            stdevP = standard_deviation(Precision)
            stdevR = standard_deviation(Recall)

        print('----------------------')
        print('Accuracy: ', average_value(Accuracy), ' ', stdevA)
        print('Precision: ', average_value(Precision), ' ', stdevP)
        print('Recall: ', average_value(Recall), ' ', stdevR)
        print('Area under ROC:', area_under_curve)


# easy to call function from jupyter notebook
def logreg(data_path: str, lamba: int, use_cross_validation: bool = True):
    path = os.path.normpath(data_path)
    base = os.path.basename(path)
    data = parse_c45(base, path)
    new_data, y = data.unzip()
    new_data['label'] = y

    Accuracy = []
    Precision = []
    Recall = []

    if use_cross_validation:
        folds = cv_split(data, 5)
        for i in range(0, len(folds)):
            cv_data = build_cv_data(folds, i)
            testing_data = folds[i]
            testing_data, y = testing_data.unzip()
            testing_data['label'] = y
            cv_data, y = cv_data.unzip()
            cv_data['label'] = y
            classifier = LogReg(cv_data, lamba)
            classifier.train()
            predicted_labels, predicted_confidence = classifier.predict(testing_data.drop("label", axis=1))
            Accuracy.append(accuracy(testing_data.label, predicted_labels))
            Precision.append(precision(testing_data.label, predicted_labels))
            Recall.append(recall(testing_data.label, predicted_labels))
        area_under_curve = auc(predicted_labels, predicted_confidence)
        evaluate_logreg(Accuracy, Precision, Recall, area_under_curve)

    else:
        test, train = np.array_split(new_data, 2)
        test.reset_index(inplace=True, drop=True)
        train.reset_index(inplace=True, drop=True)
        classifier = LogReg(train, lamba)
        classifier.train()
        predicted_labels, predicted_confidence = classifier.predict(test.drop("label", axis=1))
        Accuracy.append(accuracy(test.label, predicted_labels))
        Precision.append(precision(test.label, predicted_labels))
        Recall.append(recall(test.label, predicted_labels))
        area_under_curve = auc(predicted_labels, predicted_confidence)
        evaluate_logreg(Accuracy, Precision, Recall, area_under_curve)


# main function to call from terminal
if __name__ == '__main__':
    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a Naive Bayes algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('lamba', metavar='LAMBDA', type=float,
                        help='A constant for learning')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.set_defaults(cv=True)
    args = parser.parse_args()
    data_path = os.path.expanduser(args.path)
    lamba = args.lamba
    use_cross_validation = args.cv
    logreg(data_path, lamba, use_cross_validation)