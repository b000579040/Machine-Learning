# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('ex2data1.txt', names=['exam1', 'exam2', 'admitted'])


# x1 = pd.read_csv('ex2data1.txt', usecols=[0])
# x2 = pd.read_csv('ex2data1.txt', usecols=[1])
# y = pd.read_csv('ex2data1.txt', usecols=[2])

def get_x(df):  # 读取特征
    #     use concat to add intersect feature to avoid side effect
    #     not efficient for big dataset though
    #     """
    ones = pd.DataFrame({'ones': np.ones(len(df))})  # ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并
    return data.iloc[:, :-1]  # 这个操作返回 ndarray,不是矩阵


def get_y(df):  # 读取标签
    #     '''assume the last column is the target'''
    return np.array(df.iloc[:, -1])  # df.iloc[:, -1]是指df的最后一列


def normalize_feature(df):
    #     """Applies function along input axis(default 0) of DataFrame."""
    return df.apply(lambda column: (column - column.mean()) / column.std())


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, x, y):
    ''' cost fn is -l(theta) for you to minimize'''
    print(-y * np.log(sigmoid(x @ theta)), '\n', np.shape(-y * np.log(sigmoid(x @ theta))), '\n',
          type(-y * np.log(sigmoid(x @ theta))), '\n', np.shape(np.mat(-y * np.log(sigmoid(x @ theta)))), '\n',
          np.mat(-y * np.log(sigmoid(x @ theta))), '\n', np.reshape(np.mat(-y * np.log(sigmoid(x @ theta))), (2, 50)))
    # print((1 - y) * np.log(1 - sigmoid(x @ theta)))
    return np.mean(-y * np.log(sigmoid(x @ theta)) - (1 - y) * np.log(1 - sigmoid(x @ theta)))


def gradient(theta, x, y):
    #     '''just 1 batch gradient'''
    return (1 / len(x)) * x.T @ (sigmoid(x @ theta) - y)


def Decsent(theta, alpha, x, y):
    # diff = np.dot(x, theta) - y
    # gradient = (1 / len(x)) * x.T @ (sigmoid(x @ theta) - y)
    # theta = theta - alpha * gradient
    # i = 1
    # while np.all(np.fabs(gradient) > 1e-5):  # 判断梯度精准度
    #     diff = np.dot(x, theta) - y
    #     gradient = (1 / len(x)) * x.T @ (sigmoid(x @ theta) - y)
    #     theta = theta - alpha * gradient
    #     i = i + 1
    # print(i)
    # print(gradient)

    for i in range(1, 1000):
        diff = np.dot(x, theta) - y
        gradient = (1. / len(x)) * x.T @ (sigmoid(x @ theta) - y)
        theta = theta - alpha * gradient

    print(gradient)
    print(theta)
    return


# X @ theta与X.dot(theta)等价


theta = np.zeros(3)
x = get_x(data)
y = get_y(data)
alpha = 0.001
# Decsent(theta, alpha, x, y)
cost_function(theta, x, y)
