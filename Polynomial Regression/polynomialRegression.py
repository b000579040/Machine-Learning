import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

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

x1 = np.ones(m)
X = np.hstack((np.mat(x1).T, x))
# 正规方程公式 theta = (A.T @ A)-1 @ A.T @ Y
# theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

theta_Descent = np.array([1, 1]).reshape(2, 1)

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


theta = gradientDescent(X, y, theta_Descent, alpha)

X_plot = np.linspace(-3, 3, 1000).reshape(-1, 1)
y_plot = np.dot(X_plot, theta[1].T) + theta[0]

plt.plot(X_plot, y_plot)
plt.plot(x, y, 'b.')
plt.xlabel('X')
plt.ylabel('y')

plt.show()
# plt.savefig('regu-2.png', dpi=200)
