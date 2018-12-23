#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np;
import matplotlib.pyplot as plt;

# with open('ex1data1.txt', 'r') as f:
#     print(f.read())

dateSet = np.loadtxt('ex1data1.txt', delimiter=',')

x0 = dateSet[:, 0]
y0 = dateSet[:, 1]

m = x0.size

x1 = np.ones(m)

x = np.vstack((x1, x0)).T
y = y0.reshape(m, 1)

alpha = 0.01


def gradient_function(theta, X, y):
    diff = np.dot(X, theta) - y
    return (1. / m) * np.dot(np.transpose(X), diff)


def gradient_descent(X, y, alpha):
    theta = np.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, X, y)
    i = 1
    while np.all(np.absolute(gradient) > 1e-5):
        i = i + 1
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
    print(i)
    return theta


optimal = gradient_descent(x, y, alpha)
print('optimal:', optimal)
