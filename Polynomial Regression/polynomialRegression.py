import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from LinearRegression.linearRegression import gradientDescent
from sklearn.linear_model import LinearRegression

data = np.array([[-2.95507616, 10.94533252],
                 [-0.44226119, 2.96705822],
                 [-2.13294087, 6.57336839],
                 [1.84990823, 5.44244467],
                 [0.35139795, 2.83533936],
                 [-1.77443098, 5.6800407],
                 [-1.8657203, 6.34470814],
                 [1.61526823, 4.77833358],
                 [-2.38043687, 8.51887713],
                 [-1.40513866, 4.18262786]])
m = data.shape[0]  # 样本大小
x = data[:, 0].reshape(-1, 1)  # 将array转换成矩阵
X = x
y = data[:, 1].reshape(-1, 1)
plt.plot(X, y, "b.")
np.ones(m)
plt.xlabel('X')
plt.ylabel('y')

# lin_reg = LinearRegression()
# # lin_reg.fit(X, y)
# # print(lin_reg.intercept_, lin_reg.coef_)  # [ 4.97857827] [[-0.92810463]]

x1 = np.ones(m)
X = np.vstack((np.mat(x1).T, X))
# 正规方程公式 theta = (A.T @ A)-1 @ A.T @ Y
theta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))

X_plot = np.linspace(-3, 3, 1000).reshape(-1, 1)
y_plot = x * theta[0] + theta[1]

plt.plot(X_plot, y_plot, 'r-')
plt.plot(X, y, 'b.')
plt.xlabel('X')
plt.ylabel('y')

plt.show()
# plt.savefig('regu-2.png', dpi=200)
