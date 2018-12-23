#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
import pandas as pd

data = pd.read_csv("ex1data.txt")
dateSet = np.loadtxt('ex1data1.txt', delimiter=',')

x0 = dateSet[:, 0]
y0 = dateSet[:, 1]

m = x0.size

x1 = np.ones(m)

x = np.vstack((x1, x0)).T
y = y0.reshape(m, 1)

# 正规方程公式 theta = (A.T @ A)-1 @ A.T @ Y
print(np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y)))
