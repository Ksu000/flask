import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm
from sklearn import datasets

from joblib import dump, load

mnist_train = pd.read_csv("data/mnist_train.csv", header=None)
mnist_test = pd.read_csv("data/mnist_test.csv", header=None)

train_data = mnist_train.values[:, 1:]
test_data = mnist_test.values[:, 1:]

train_label = mnist_train.values[:, 0]
test_label = mnist_test.values[:, 0]

kn_classifier = KNeighborsClassifier(n_jobs=-1)

mlp_classifier = MLPClassifier(hidden_layer_sizes=(512,),verbose=True)
mlp_classifier = mlp_classifier.fit(train_data, train_label)

# https://ru.stackoverflow.com/questions/966281/%D0%9E%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%BD%D0%B0%D1%8F-%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F-%D1%81%D0%B5%D1%82%D1%8C-%D0%BD%D0%B5-%D0%BC%D0%BE%D0%B6%D0%B5%D1%82-%D1%80%D0%B0%D1%81%D0%BF%D0%BE%D0%B7%D0%BD%D0%B0%D1%82%D1%8C-%D0%BC%D0%BE%D0%B8-%D1%80%D1%83%D0%BA%D0%BE%D0%BF%D0%B8%D1%81%D0%BD%D1%8B%D0%B5-%D1%87%D0%B8%D1%81%D0%BB%D0%B0-%D0%BD%D0%BE-%D0%BF%D1%80%D0%B5%D0%BA%D1%80%D0%B0%D1%81%D0%BD%D0%BE
