import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from util import *

np.random.seed(12345)

class ELM:
    def __init__(self, num_hidden_nodes):
        self.num_hidden_nodes = num_hidden_nodes
        self.output_weights = None
        self.weight_matrix = None
        self.bias = np.random.rand(num_hidden_nodes)
        self.H = None

    def train(self, data):
        labels = data.label
        data = data.drop("label", axis=1)
        n = len(data.columns)
        self.weight_matrix = np.random.rand(self.num_hidden_nodes, n)
        h = self.hidden_layer_output(data)
        self.H = h
        self.output_weights = np.dot(labels, np.linalg.pinv(h))

    def update(self, example, m0):
        y = example[-1]
        example = example[0:-1]
        h = np.asmatrix(self.hidden_layer_output(example)).transpose()
        hht = np.dot(h, h.transpose())
        top = np.dot(np.dot(m0, hht), m0)
        bot = 1 + np.dot(np.dot(h.transpose(), m0), h)
        x = m0 - top / bot
        m0 = x
        self.output_weights = np.asmatrix(self.output_weights).transpose() + np.dot(np.dot(m0, h), (
                    y - np.dot(h.transpose(), self.output_weights)))
        return m0

    def predict(self, data):
        h = self.hidden_layer_output(data)
        return self.calculate_output(h)

    def hidden_layer_output(self, data):
        return sigmoid((np.dot(self.weight_matrix, data.transpose()).transpose() - self.bias).transpose())

    def calculate_output(self, hidden_output):
        return np.dot(hidden_output.transpose(), self.output_weights)


def sigmoid(x):
    return (1 + np.e ** -x) ** -1


def update_nominals(data):
    nominal_data = data.select_dtypes(exclude="float")
    labels = nominal_data.label
    nominal_data = nominal_data.drop("label", axis=1)
    float_data = data.select_dtypes(include="float")
    x, y = nominal_data.shape
    noise_matrix = pd.DataFrame(np.random.rand(x, y))
    noise_matrix.columns = nominal_data.columns
    nominal_data -= noise_matrix
    return float_data.join(nominal_data.join(labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ELM')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('type', metavar='TYPE', type=int, help='which dataset to run on')
    args = parser.parse_args()
    data = pd.read_csv(args.path)
    data = data.sample(len(data), axis=0).reset_index(drop=True)
    data = update_nominals(data)

    if (args.type == 1):  # loan
        # for i in range(len(data)):
        # if data.label[i] == 0:
        # data.label[i] = -1
        None
    elif (args.type == 2):  # iris
        for i in range(len(data)):
            if data.label[i] == "Iris-setosa":
                data.label[i] = 1
            else:
                data.label[i] = 2
    elif (args.type == 3):  # heart
        None

    train, updating, test = np.array_split(data, 3)
    num_examples = np.array([0])
    acc = np.array([0])
    classifier = ELM(2 * len(data.columns))
    start = time.time()
    classifier.train(train)
    prediction = classifier.predict(test.drop("label", axis=1))
    l = len(train)
    num_examples = np.append(num_examples, l)
    acc = np.append(acc, sum(np.round(prediction.astype("float")) ==
                             test.label) / len(test))
    m0 = np.linalg.inv(np.dot(classifier.H, classifier.H.transpose()))
    for i in range(len(updating)):
        example = updating.iloc[i, :]
        m0 = classifier.update(example, m0)
        acc = np.append(acc, sum(np.round(prediction.astype("float")) == test.label) / len(test))
        num_examples = np.append(num_examples, l + i)
    end = time.time()
    print(end - start)
    print(1 - sum(test.label) / len(test))
    plt.plot(num_examples, acc)
    plt.title("Learning Curve")
    plt.xlabel("Number of Examples")
    plt.ylabel("Accuracy")
    plt.show()
