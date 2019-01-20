#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np;
import matplotlib.pyplot as plt;


with open('ex1data1.txt', 'r') as f:
    print(f.read())

dateSet = np.loadtxt('ex1data1.txt', delimiter=',')

x0 = dateSet[:, 0]
y0 = dateSet[:, 1]

m = x0.size

x1 = np.ones(m)

x = np.vstack((x1, x0)).T
y = y0.reshape(m, 1)

theta = np.array([1, 1]).reshape(2, 1)

alpha = 0.01


def gradientDescent(x, y, theta, alpha):
    # 迭代i次，退出循环
    # for i in range(1, 5000):
    #     diff = np.dot(x, theta) - y
    #     gradient = (1. / m) * np.dot(np.transpose(x), diff)
    #     theta = theta - alpha * gradient
    m = x.size
    diff = np.dot(x, theta) - y
    gradient = (1. / m) * np.dot(np.transpose(x), diff)
    theta = theta - alpha * gradient
    i = 1
    while np.all(np.fabs(gradient) > 1e-5):  # 判断梯度精准度
        diff = np.dot(x, theta) - y
        gradient = (1. / m) * np.dot(np.transpose(x), diff)
        theta = theta - alpha * gradient
        i = i + 1
    print(i)
    print('gradient', gradient)
    print('theta', theta)

    return theta

theta = gradientDescent(x, y, theta, alpha)
# np.set_printoptions(precision=5)
print(theta)
