import argparse
import os.path
import numpy as np
import pandas as pd

from util import *

from typing import Sequence

from sting.classifier import Classifier
from sting.data import AbstractDataSet, parse_c45, Feature


# this is the node class that will be the building blocks of the tree
class Node:
    def __init__(self, edge):
        self.children = []
        self.edge = edge
        self.name = None

    def add_child(self, child):
        self.children.append(child)


class DecisionTree(Classifier):

    def __init__(self, data: AbstractDataSet, depth_limit: int, use_cv: bool, use_ig: bool = True):
        self.tree = Node("root")
        self.depth_limit = depth_limit
        self.data = data
        self.use_cv = use_cv
        self.use_ig = use_ig

    def train(self) -> None:
        """
        You will implement this method.
        :param data: Labelled dataset with which to train the decision tree.
        """
        df, y = self.data.unzip()
        df['label'] = y
        tests = get_tests(self.data)
        height = min(self.depth_limit, len(tests))
        id3(df, tests, height, self.tree, self.use_ig)

    def predict(self, data: AbstractDataSet) -> Sequence[int]:
        """
        You will implement this method.
        :param data: Unlabelled dataset to make predictions
        :return: Predictions as a sequence of 0s and 1s. Any sequence (list, numpy array, generator, pandas Series) is
        acceptable.
        """
        df, y = data.unzip()
        predictions = []
        for i in range(0, len(df)):
            predictions.append(self.classify_example(df.iloc[i], self.tree))
        return predictions

    # first research extension essentially trying to minimize gini impurity
    def best_test_extension(self, data, tests):
        val = float('inf')
        best = None
        for test in tests:
            tot = 0
            splits = split_data_on_test(data, test)
            for dset in splits:
                p = sum(dset.label == 1) / len(dset)
                tot += p * (1 - p)
            if tot / len(splits) < val:
                val = tot
                best = test
        return best

    # second research extension is finding the best attribute if guessing 50/50 is bad
    def best_test_extension_two(self, data, tests):
        val = 0
        best = None
        for test in tests:
            tot = 0
            for dset in split_data_on_test(data, test):
                p = sum(dset.label == 1) / len(dset)
                tot += np.abs(2 * p - 1)
            if tot >= val:
                val = tot
                best = test
        return best

    # given one example this will traverse the tree and classify it
    def classify_example(self, example, root):
        tracer = root
        if len(root.children) == 0:
            return root.name
        if isinstance(tracer.name, tuple):
            value = example[tracer.name[0]] <= tracer.name[1]
        else:
            value = example[tracer.name]
        for child in tracer.children:
            if child.edge == value:
                return self.classify_example(example, child)


# will get the entropy of feature in a dataset
def nominal_entropy(data, feature):
    tot = 0
    for value in np.unique(data[feature]):
        p = sum(data[feature] == value) / len(data)
        tot += (p * np.log2(p))
    return -1 * tot


# gets the entropy of a continuous test in a dataset
def continuous_entropy(data, test):
    return nominal_entropy(data[data[test[0]] <= test[1]], "label") + nominal_entropy(
        data[data[test[0]] > test[1]],
        "label")


# a generic get entropy function that handles the type of feature
def entropy(data, test):
    if isinstance(test, tuple):
        return continuous_entropy(data, test)
    else:
        return nominal_entropy(data, test)


# will create subsets of the data
def split_data_on_test(data, test):
    # print(isinstance(test,tuple))
    datasets = []
    if isinstance(test, tuple):
        datasets.append(data[data[test[0]] <= test[1]])
        datasets.append(data[data[test[0]] > test[1]])
    else:
        for value in np.unique(data[test]):
            datasets.append(data[data[test] == value])

    return datasets


# gets the information gain of a test
def information_gain(data, test):
    ig = entropy(data, "label")
    for frame in split_data_on_test(data, test):
        ig -= entropy(frame, "label") * len(frame) / len(data)
    return ig


# gets gain ratio of a test on a dataset
def gain_ratio(data, test):
    h = entropy(data, test)
    if h == 0:
        return float('inf')
    else:
        return information_gain(data, test) / h


# will return the best test depending on the metric used
def get_best_test(data, tests, ig):
    if ig:
        f = information_gain
    else:
        f = gain_ratio
    best = None
    val = 0
    for test in tests:
        if f(data, test) >= val:
            val = f(data, test)
            best = test
    return best


# the id3 alg to build the trained DT
def id3(data, tests, height, root, use_ig):
    if len(tests) == 0 or height == 0:  # if no more tests or limit reached
        root.name = data.mode()["label"][0]  # pick maximum likelihood one
        return
    else:
        test = get_best_test(data, tests, use_ig)  # get the best test to split on
        if information_gain(data, test) == 0:
            root.name = data.mode()["label"][0]
            return
        else:
            tests.remove(test)
            root.name = test
            if type(test) is tuple:
                l = Node(True)
                r = Node(False)
                root.add_child(l)
                id3(data[data[test[0]] <= test[1]], tests.copy(), height - 1, l, use_ig)
                root.add_child(r)
                id3(data[data[test[0]] > test[1]], tests.copy(), height - 1, r, use_ig)
            else:
                for value in np.unique(data[test]):
                    n = Node(value)
                    root.add_child(n)
                    id3(data[data[test] == value], tests.copy(), height - 1, n, use_ig)


# gets the continuous splits of a continuous feature in a dataset
def get_continuous_splits(data, feature):
    sorted_data = list(zip(data[feature], data['label']))
    sorted_data.sort()
    splits = []
    for i in range(1, len(sorted_data)):
        if sorted_data[i][0] == sorted_data[i - 1][0]:
            continue
        if sorted_data[i][1] != sorted_data[i - 1][1]:
            splits.append((feature, (sorted_data[i][0] + sorted_data[i - 1][0]) / 2))
    return splits


# gets all the tests needed from a dataset
def get_tests(data: AbstractDataSet):
    features = data.schema
    df, y = data.unzip()
    df["label"] = y
    tests = []
    for feature in features:
        if feature.ftype == Feature.Type.CONTINUOUS:
            tests.extend(get_continuous_splits(df, feature.name))
        elif feature.ftype == Feature.Type.NOMINAL or feature.Type == Feature.Type.BINARY:
            tests.append(feature.name)
    return tests


def evaluate_dtree(dtree: DecisionTree, dataset: AbstractDataSet):
    """
    You will implement this method.
    Given a trained decision tree and labelled dataset, Evaluate the tree and print metrics.
    :param dtree: Trained decision tree
    :param dataset: Testing set
    """
    predictions = dtree.predict(dataset)
    acc = accuracy(dataset.unzip()[1].values, predictions)
    print('----------------------')
    print('Accuracy: ', acc)
    print('Size:', size_of_tree(dtree.tree))
    print('Maximum Depth:', find_max_depth(dtree.tree))
    print('First Feature:', dtree.tree.name)

    return


def dtree(data_path: str, tree_depth_limit: int, use_cross_validation: bool = True, use_information_gain: bool = True):
    """
    It is highly recommended that you make a function like this to run your program so that you are able to run it
    easily from a Jupyter notebook.
    :param data_path: The path to the data.
    :param tree_depth_limit: Depth limit of the decision tree
    :param use_cross_validation: If True, use cross validation. Otherwise, run on the full dataset.
    :param information_gain: If true, use information gain as the split criterion. Otherwise use gain ratio.
    :return:
    """

    path = os.path.normpath(data_path)
    base = os.path.basename(path)
    data = parse_c45(base, path)

    totalAccuracy = 0.0

    if use_cross_validation:
        folds = cv_split(data, 5)
        for i in range(0, len(folds)):
            cv_data = build_cv_data(folds, i)
            testing_data = folds[i]
            cv_tree = DecisionTree(cv_data, tree_depth_limit, use_cross_validation, use_information_gain)
            cv_tree.train()
            predicted_labels = cv_tree.predict(testing_data)
            features, actual_labels = testing_data.unzip()
            totalAccuracy += accuracy(actual_labels.values, predicted_labels)

            evaluate_dtree(cv_tree, testing_data)
        totalAccuracy /= len(folds)
        print('Total average accuracy: ', totalAccuracy)

    tree = DecisionTree(data, tree_depth_limit, use_cross_validation, use_information_gain)
    tree.train()

    evaluate_dtree(tree, data)

    return tree


# finds the maximum depth of a tree
def find_max_depth(root):
    if len(root.children) == 0:
        return 0
    maxdepth = 0
    for child in root.children:
        maxdepth = max(maxdepth, find_max_depth(child))
    return maxdepth + 1


# finds the number of nodes in a tree
def size_of_tree(root):
    sum = 0
    if len(root.children) == 0:
        return 1
    for child in root.children:
        sum += 1 + size_of_tree(child)
    return sum


if __name__ == '__main__':
    """
    THIS IS YOUR MAIN FUNCTION. You will implement the evaluation of the program here. We have provided argparse code
    for you for this assignment, but in the future you may be responsible for doing this yourself.
    """

    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a decision tree algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('depth_limit', metavar='DEPTH', type=int,
                        help='Depth limit of the tree. Must be a non-negative integer. A value of 0 sets no limit.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.add_argument('--use-gain-ratio', dest='gain_ratio', action='store_true',
                        help='Use gain ratio as tree split criterion instead of information gain.')
    parser.set_defaults(cv=True, gain_ratio=False)
    args = parser.parse_args()

    # If the depth limit is negative throw an exception
    if args.depth_limit < 0:
        raise argparse.ArgumentTypeError('Tree depth limit must be non-negative.')

    # You can access args with the dot operator like so:
    data_path = os.path.expanduser(args.path)
    tree_depth_limit = args.depth_limit
    use_cross_validation = args.cv
    use_information_gain = not args.gain_ratio

    tree = dtree(data_path, tree_depth_limit, use_cross_validation, use_information_gain)