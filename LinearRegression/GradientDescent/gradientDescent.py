#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np

# Size of the points dataset.  数据集合长度
m = 12

# The Learning Rate alpha.  学习速率
alpha = 0.01

# 训练数据
# Points x-coordinate and dummy value (x0, x1). x坐标数据
X0 = np.ones((m, 1))
# X1 = np.arange(1, m + 1).reshape(m, 1)
X1 = np.array([1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 8, 10]).reshape(m, 1)
X = np.hstack((X0, X1))

# Points y-coordinate   y坐标数据
# y = np.array([
#     3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
#     11, 13, 13, 16, 17, 18, 17, 19, 21
# ]).reshape(m, 1)

y = np.array([
    -890, -1411, -1560, -2220, -2091, -2878, -3537, -3268, -3920, -4163, -5471, -5157
]).reshape(m, 1)


# print("X0\n", X0, "\n")
# print("X1\n", X1, "\n")
#
# print("X\n", X, "\n")


# print(y)


# 报错时候调用
# def error_function(theta, X, y):
#     '''Error function J definition.'''
#     diff = np.dot(X, theta) - y
#     return (1. / 2 * m) * np.dot(np.transpose(diff), diff)


# 梯度下降算法中 求最大坡度J
def gradient_function(theta, X, y):
    '''Gradient of the function J definition.'''
    diff = np.dot(X, theta) - y
    return (1. / m) * np.dot(np.transpose(X), diff)


# 执行梯度迭代
def gradient_descent(X, y, alpha):
    '''Perform gradient descent.'''
    theta = np.array([1, 1]).reshape(2, 1)

    gradient = gradient_function(theta, X, y)

    while not np.all(np.absolute(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
    return theta


optimal = gradient_descent(X, y, alpha)

print('optimal:', optimal)
# print('error function:', error_function(optimal, X, y))
