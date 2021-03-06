import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt;


# 加载数据
def loadData(file):
    mat = sio.loadmat(file)
    data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
    data.insert(2, 'Cluster', 0)
    return data


# 计算欧式距离
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))  # 求两个向量之间的距离


# 随机初始聚点
def randCent(data, k):
    max = data.max(0)  # 数据集最大坐标
    cent = pd.DataFrame(columns=['X1', 'X2'])
    for i in range(k):
        row = {'X1': np.random.uniform(0, max[0]), 'X2': np.random.uniform(0, max[1]),
               'Cluster': i + 1}  # 随机初始化点在0到最大坐标之间
        cent = cent.append(row, ignore_index=True)
    return cent


# 数据集找对应的聚点
def findCluster(data, cent):
    for i in range(data.shape[0]):  # 遍历每个数据集
        dis = []
        for j in range(cent.shape[0]):  # 对比每个聚点欧式距离
            dis.append(distEclud(np.array([data.get('X1')[i], data.get('X2')[i]]),
                                 np.array([cent.get('X1')[j], cent.get('X2')[j]])))
            data.loc[i, 'Cluster'] = dis.index(min(dis)) + 1
    return data


# 更新聚点位置
def updateCluster(data, k):
    newCent = pd.DataFrame(columns=['X1', 'X2', 'Cluster'])
    for i in range(k):  # 遍历每个类别
        row = {'X1': data[data.Cluster == i + 1].mean().get('X1'), 'X2': data[data.Cluster == i + 1].mean().get('X2'),
               'Cluster': i + 1}  # 根据分配给同一个聚类的数据集的平均值确定新的聚点（数据中心点）
        newCent = newCent.append(row, ignore_index=True)
    return newCent


# 聚点代价（L1聚点距离）
def cost(cent, newCent):
    cost = np.fabs(newCent - cent).drop('Cluster', 1)
    return cost


data = loadData('ex7data2.mat')

cent = randCent(data, 3)
# cent = pd.DataFrame({'X1': [2, 2, 6], 'X2': [5, 1, 3], 'Cluster': [1.0, 2.0, 3.0]})
data = findCluster(data, cent)
newCent = updateCluster(data, 3)
i = 1
while np.all(cost(cent, newCent) > 1e-7):  # 判断误差精准度
    cent = newCent
    data = findCluster(data, cent)
    newCent = updateCluster(data, 3)
    i = i + 1
print(i)

plt.scatter(data[data.Cluster == 1].get('X1'), data[data.Cluster == 1].get('X2'), c='r')
plt.scatter(data[data.Cluster == 2].get('X1'), data[data.Cluster == 2].get('X2'), c='b')
plt.scatter(data[data.Cluster == 3].get('X1'), data[data.Cluster == 3].get('X2'), c='g')
# plt.scatter(cent.get('X1'), cent.get('X2'), color='k')
plt.show()
