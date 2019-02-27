import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def and_function(x1, x2):
    theta = np.matrix([-30, 20, 20])
    x = np.matrix([1, x1, x2])
    # print(sigmoid(np.dot(theta, x.T)))
    return sigmoid(np.dot(theta, x.T))


def or_function(x1, x2):
    theta = np.matrix([-10, 20, 20])
    x = np.matrix([1, x1, x2])
    # print(sigmoid(np.dot(theta, x.T)))
    return sigmoid(np.dot(theta, x.T))


def nagtion_function(x):
    theta = np.matrix([10, -20])
    x = np.matrix([1, x])
    # print(sigmoid(np.dot(theta, x.T)))
    return sigmoid(np.dot(theta, x.T))


def XOR_function(x1, x2):
    # print(and_function(x1, x2))
    # print(and_function(nagtion_function(x1), nagtion_function(x2)))
    return or_function(and_function(x1, x2), and_function(nagtion_function(x1), nagtion_function(x2)))


# print(XOR_function(0, 1))
# print(XOR_function(1, 0))
# print(XOR_function(1, 1))
# print(XOR_function(0, 0))

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[1, 0, 0, 1]]).T

l1 = sigmoid(X)

