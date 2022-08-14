import argparse
import os.path
import statistics
import warnings

import numpy as np
import pandas as pd

from util import *

from sting.classifier import Classifier
from sting.data import AbstractDataSet, parse_c45, Feature
warnings.filterwarnings('ignore')

class NBayes(Classifier):

    #Constructor for nabayes
    def __init__(self, num_bins, m, dataset, use_cv: bool, research: bool = False):
        self.table = None
        self.num_bins = num_bins
        self.m = m
        self.data = dataset
        self.use_cv = use_cv
        self.bin_sizes = {}
        self.research = research

    # trains the data
    def train(self) -> None:
        df, y = self.data.unzip()
        df['label'] = y
        if self.research:
            self.data = ranged_bins(df, self.num_bins, self)
        else:
            self.data = split_into_bins(df, self.num_bins)
        self.table = table(self.data, self.m)

    #predicts the data
    def predict(self, data):
        data, y = data.unzip()
        if self.research:
            data = r_format_data(data,self)
        else:
            data = split_into_bins(data, self.num_bins)
        pos = np.array([0.0 for x in range(0, len(data))])
        neg = np.array([0.0 for x in range(0, len(data))])
        conf = np.array([0.0 for x in range(0, len(data))])
        for example in range(0, len(data)):
            for feature in data.columns:
                try:
                    v1 = self.table[feature][data[feature][example]][1]
                    v0 = self.table[feature][data[feature][example]][0]
                    p_1 = np.log2(v1)
                    p_0 = np.log2(v0)
                except KeyError:
                    p_1 = np.log2(self.table["pos_prior"])
                    p_0 = np.log2(self.table["neg_prior"])
                if p_1 == -float('inf'):
                    pos[example] = 0
                    continue
                else:
                    pos[example] += p_1
                if p_0 == -float('inf'):
                    neg[example] = 0
                else:
                    neg[example] += p_0
        for i in range(0, len(data)):
            p = pos[i] + np.log2(self.table["pos_prior"])
            n = neg[i] + np.log2(self.table["neg_prior"])
            if p >= n:
                conf[i] = p
                pos[i] = 1
            else:
                conf[i] = n
                pos[i] = 0
        return pos, conf


# splits the data into bina
def split_into_bins(data, n):
    df1 = data.select_dtypes(include=float)
    df2 = data.select_dtypes(exclude=float)
    for feature in df1.columns:
        df1[feature] = (np.floor(df1[feature] - min(df1[feature])) % n) + 1
    result = df1.join(df2)
    result.reset_index(inplace=True, drop=True)
    return result

# splits into bins for research extension
def ranged_bins(data, n, classifier:NBayes):
    df1 = data.select_dtypes(include=float)
    df2 = data.select_dtypes(exclude=float)
    for feature in df1.columns:
        min = np.min(df1[feature].values)
        df1[feature] = df1[feature] - min
        max = np.max(df1[feature])
        binsize = np.ceil(max / n)
        df1[feature] = np.floor(df1[feature] / binsize)
        classifier.bin_sizes[feature] = binsize
    df2 = df2.join(df1)
    df2.reset_index(inplace=True, drop=True)
    return df2

# takes dataframe we want to predict and puts them into bins
def r_format_data(data, classifier:NBayes):
    for col in data.columns.values:
        try:
            binsize = classifier.bin_sizes[col]
            data[col] = np.floor(data[col] / binsize)
        except KeyError:
            pass
    return data

# prints the likelyhood table for naive bayes
def likeyhood_table(self):

    for feature in self.data.columns:
        if (feature != 'index') and (feature != 'weight') and (feature != 'label'):
            values = np.unique(self.data[feature])
            break

    negtab = pd.DataFrame(columns=values)

    for features in self.data.columns:
        if (features != 'index') and (features != 'weight') and (features != 'label'):
            new_row = {'Feature': features}
            for value in values:
                new_row.update({value: self.table.get(features).get(value)[0]})

            negtab = negtab.append(new_row, ignore_index=True)

    print("Negative Label Likelyhood Table")
    print(negtab)

    print("-------------------------------------------------------")

    postab = pd.DataFrame(columns=values)
    for features in self.data.columns:
        if (features != 'index') and (features != 'weight') and (features != 'label'):
            new_row = {'Feature': features}
            for value in values:
                new_row.update({value: self.table.get(features).get(value)[1]})
            postab = postab.append(new_row, ignore_index=True)

    print("Positive Label Likelyhood Table")
    print(postab)
    print("-------------------------------------------------------")


# takes a pandas dataframe and m value and returns a dictionary col -> value -> prob
def table(data, m):
    features = data.drop(['weight', 'label'], axis=1)
    tab = {x: None for x in features}
    pos = data[data.label == 1]
    neg = data[data.label == 0]
    for feature in features:
        values = np.unique(data[feature])
        tab[feature] = {x: None for x in values}
        p = 1 / len(values)
        for value in values:
            p_1 = (len(pos[pos[feature] == value]) + (m * p)) / (len(pos) + m)
            p_0 = (len(neg[neg[feature] == value]) + (m * p)) / (len(neg) + m)
            tab[feature][value] = (p_0, p_1)
    tab['pos_prior'] = len(pos) / len(data)
    tab['neg_prior'] = len(neg) / len(data)
    return tab


def standard_deviation(arr: [float]):
    return statistics.stdev(arr)


def average_value(arr: [float]):
    return sum(arr) / len(arr)


# prints out the metrics of our functions
def evaluate_nbayes(Accuracy: [float], Precision: [float], Recall: [float], area_under_curve: float):
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


# function to easily call from the notebook
def nbayes(data_path: str, number_bins: int, m_estimate: int, use_cross_validation: bool = True, use_research: bool = False):
    path = os.path.normpath(data_path)
    base = os.path.basename(path)
    data = parse_c45(base, path)
    Accuracy = []
    Precision = []
    Recall = []
    if use_cross_validation:
        folds = cv_split(data, 5)
        for i in range(0, len(folds)):
            cv_data = build_cv_data(folds, i)
            testing_data = folds[i]
            classifier = NBayes(number_bins, m_estimate, cv_data, use_cross_validation, use_research)
            classifier.train()
            predicted_labels, predicted_confidence = classifier.predict(testing_data)
            features, actual_labels = testing_data.unzip()
            Accuracy.append(accuracy(actual_labels.values, predicted_labels))
            Precision.append(precision(actual_labels.values, predicted_labels))
            Recall.append(recall(actual_labels.values, predicted_labels))
        #area_under_curve = auc(actual_labels.values,predicted_labels)
        evaluate_nbayes(Accuracy, Precision, Recall, 0)

    else:
        classifier = NBayes(number_bins, m_estimate, data, use_cross_validation, use_research)
        classifier.train()
        predicted_labels, predicted_confidence = classifier.predict(data)
        features, actual_labels = data.unzip()
        Accuracy.append(accuracy(actual_labels.values, predicted_labels))
        Precision.append(precision(actual_labels.values, predicted_labels))
        Recall.append(recall(actual_labels.values, predicted_labels))
        #area_under_curve = auc(predicted_labels, predicted_confidence)
        evaluate_nbayes(Accuracy, Precision, Recall, 0)


# main function to call from terminal
if __name__ == '__main__':

    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a Naive Bayes algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('bins', metavar='BINS', type=int,
                        help='Number of bins to divide continuous features into')
    parser.add_argument('estimate', metavar='ESTIMATE', type=float,
                        help='A nonnegative integer m for the m-estimate. If this value is negative, Laplace smoothing will be used.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.add_argument('--research', dest='research', action='store_true',
                        help='Enables research extension for this project.')

    parser.set_defaults(cv=True, research=False)
    args = parser.parse_args()

    # If the number of bins is less than 2, throw an exception
    if args.bins < 2:
        raise argparse.ArgumentTypeError('Number of bins must be at least 2')

    data_path = os.path.expanduser(args.path)
    number_bins = args.bins
    use_cross_validation = args.cv
    use_research = args.research
    m_estimate = args.estimate

    nbayes(data_path, number_bins, m_estimate, use_cross_validation, use_research)