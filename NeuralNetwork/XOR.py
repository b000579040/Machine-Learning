import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def and_function(x):
    theta = np.matrix([-30, 20, 20])
    x = np.mat(x)
    x1 = np.mat(np.ones(1))
    X = np.hstack((x1, x))
    print(sigmoid(np.dot(theta, X.T)))
    return sigmoid(np.dot(theta, X.T))


def or_function(x):
    theta = np.matrix([-10, 20, 20])
    x = np.mat(x)
    x1 = np.mat(np.ones(1))
    X = np.hstack((x1, x))
    print(sigmoid(np.dot(theta, X.T)))
    return sigmoid(np.dot(theta, X.T))


x = [0, 1]

and_function(x)

or_function(x)
