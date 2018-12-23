"""使用scatter()绘制散点图"""
import numpy as np
import matplotlib.pyplot as plt

# Size of the points dataset.  数据集合长度
m = 20

# 训练数据
# Points x-coordinate and dummy value (x0, x1). x坐标数据
X0 = np.ones((m, 1))
X1 = np.arange(1, m + 1).reshape(m, 1)
X = np.hstack((X0, X1))

# Points y-coordinate   y坐标数据
y = np.array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)

'''
scatter() 
x:横坐标 y:纵坐标 s:点的尺寸
'''
plt.scatter(X1, y, s=100)

# 设置图表标题并给坐标轴加上标签
plt.title('Square Numbers', fontsize=24)
plt.xlabel('Value', fontsize=14)
plt.ylabel('Square of Value', fontsize=14)

# 设置刻度标记的大小
plt.tick_params(axis='both', which='major', labelsize=14)

# 生成测试数据
x = np.linspace(1, 20, 50)
y = np.cos(x)

# 画图
plt.plot(x, y)  # 默认

plt.show()
