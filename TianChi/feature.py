import pandas as pd
import time
import os
import numpy as np

startTime = time.time()


def read_data(path):
    """
    time:刷卡时间
    lineID:地铁线路
    stationID:地铁站ID
    deviceID:地铁站刷卡设备ID
    status:0进站 1出站
    userID:用户身份ID
    payType:刷卡类型 3临时票
    """
    # 读所有原始数据文件
    files = []
    for file in os.walk(path):
        files.append(file)  # 获取文件地址

    filenames = []
    for dir_t in files:
        if dir_t[2]:
            for filename in dir_t[2]:
                filenames.append(os.path.join(dir_t[0], filename))
    filenames.sort(key=lambda x: int(x[-6:-4]))  # 文件排序

    for k in filenames:
        jan = pd.read_csv(k)  # 读数据
        mkpath = os.path.dirname(os.path.realpath(__file__)) + "/" + str(k[-14:-4])
        os.mkdir(mkpath)

        for i in range(81):
            if i == 54:
                continue
            df1 = jan[(jan.stationID == i) & (jan.status == 0)]
            df2 = jan[(jan.stationID == i) & (jan.status == 1)]

            t1 = df1['time'].map(lambda x: int(str(x[-8:-6])) * 6 + int(str(x[-5:-4])))
            t2 = df2['time'].map(lambda x: int(str(x[-8:-6])) * 6 + int(str(x[-5:-4])))

            a = t1.values[0]
            b = t2.values[0]
            num1 = 1
            num2 = 1
            re1 = []
            re2 = []

            for j in range(1, t1.shape[0]):
                if a == t1.values[j]:
                    num1 = num1 + 1
                else:
                    re1.append([a, num1])
                    a = t1.values[j]
                    num1 = 1
            for j in range(1, t2.shape[0]):
                if b == t2.values[j]:
                    num2 = num2 + 1
                else:
                    re2.append([b, num2])
                    b = t2.values[j]
                    num2 = 1

            np.savetxt(mkpath + "/arrival" + str(i) + ".txt", re1, fmt="%d")
            np.savetxt(mkpath + "/departure" + str(i) + ".txt", re2, fmt="%d")


# 将原始数据补零
def train_expand_zero(path):
    files_1 = []
    for file in os.listdir(path):
        if file[0] != ".":
            files_1.append(file)  # 获取一级文件夹内容

    files = []
    for i in range(np.shape(files_1)[0]):
        for file in os.listdir(path + "/" + files_1[i]):
            if file[0] != ".":
                files.append(path + "/" + files_1[i] + "/" + file)  # 获取二级文件夹内容

    # savePath = files[1][:11] + "new_" + files[1][11:]
    # print(files[1], savePath)

    # 将空白数据填上0
    for file in files:
        x = np.loadtxt(file)
        # print(x.shape)
        if x.shape[0] < 144:
            for i in range(1, 144 - int(x[-1][0])):
                x = np.append(x, [[int(x[-1][0]) + i, 0]], axis=0)
        for i in range(144):
            if x[i][0] > i:
                x = np.insert(x, i, [i, 0], axis=0)

        if file[-6:-5] > "99":
            file_list = list(file)
            file_list.insert(-5, "0")
            file = "".join(file_list)

        savePath = file[:5] + "new_" + file[5:]
        np.savetxt(savePath, x, fmt="%d")

    # 将数据左右的数据,扩展训练集


# 将训练集按时间扩充
def train_expend_time(path):
    files_1 = []
    for file in os.listdir(path):
        if file[0] != ".":
            files_1.append(file)  # 获取一级文件夹内容

    files = []
    for i in range(np.shape(files_1)[0]):
        for file in os.listdir(path + "/" + files_1[i]):
            if file[0] != ".":
                files.append(path + "/" + files_1[i] + "/" + file)  # 获取二级文件夹内容

    for file in files:
        x = np.loadtxt(file)

        y = []
        for i in range(144):
            if i < 38:
                y.append(x[i])
            elif (i > 37) & (i < 40):
                y.append(x[i])
                y.append([i, x[i + 1][1]])
                y.append([i, x[i + 2][1]])
            elif (i > 39) & (i < 141):
                y.append([i, x[i - 1][1]])
                y.append([i, x[i - 2][1]])
                y.append(x[i])
                y.append([i, x[i + 1][1]])
                y.append([i, x[i + 2][1]])
            elif i == 141:
                y.append([i, x[i - 1][1]])
                y.append([i, x[i - 2][1]])
                y.append(x[i])
                y.append([i, x[i + 1][1]])
            elif i == 142:
                y.append([i, x[i - 1][1]])
                y.append([i, x[i - 2][1]])
                y.append(x[i])
            elif i == 143:
                y.append(x[i])
        np.savetxt(file, y, fmt="%d")


# 将数据集添加结果

def train_expend_Y():
    path_1 = 'data/train/data_t/2019-01-01'
    path_2 = 'data/train/data_t/2019-01-08'
    path_3 = 'data/train/data_t/2019-01-15'
    path_4 = 'data/train/data_t/2019-01-22'

    files_1 = []
    files_2 = []
    files_3 = []
    files_4 = []

    for file_1 in os.listdir(path_1):  # 获取1.1内容
        if file_1[0] != ".":
            files_1.append(path_1 + "/" + file_1)

    for file_2 in os.listdir(path_2):  # 获取1.8内容
        if file_2[0] != ".":
            files_2.append(path_2 + "/" + file_2)

    for file_3 in os.listdir(path_3):  # 获取1.15内容
        if file_3[0] != ".":
            files_3.append(path_3 + "/" + file_3)

    for file_4 in os.listdir(path_4):  # 获取1.22内容
        if file_4[0] != ".":
            files_4.append(path_4 + "/" + file_4)

    file_a = []
    file_d = []
    for file in files_1:
        if file[-7:-6] == "l":
            file_a.append(file)
        else:
            file_d.append(file)
    file_a.sort(key=lambda x: int(x[-6:-4]))  # 文件排序
    file_d.sort(key=lambda x: int(x[-6:-4]))
    files_1 = []
    files_1 = file_a + file_d

    # 文件排序3
    file_a = []
    file_d = []
    for file in files_2:
        if file[-7:-6] == "l":
            file_a.append(file)
        else:
            file_d.append(file)
    file_a.sort(key=lambda x: int(x[-6:-4]))  # 文件排序
    file_d.sort(key=lambda x: int(x[-6:-4]))
    files_2 = []
    files_2 = file_a + file_d

    # 文件排序3
    file_a = []
    file_d = []
    for file in files_3:
        if file[-7:-6] == "l":
            file_a.append(file)
        else:
            file_d.append(file)
    file_a.sort(key=lambda x: int(x[-6:-4]))  # 文件排序
    file_d.sort(key=lambda x: int(x[-6:-4]))
    files_3 = []
    files_3 = file_a + file_d

    # 文件排序4
    file_a = []
    file_d = []
    for file in files_4:
        if file[-7:-6] == "l":
            file_a.append(file)
        else:
            file_d.append(file)
    file_a.sort(key=lambda x: int(x[-6:-4]))  # 文件排序
    file_d.sort(key=lambda x: int(x[-6:-4]))
    files_4 = []
    files_4 = file_a + file_d

    print(files_1[1], np.shape(files_1), np.shape(files_2), np.shape(files_3), np.shape(files_4))
    for i in range(np.shape(files_1)[0] + 1):
        if i == 54:
            continue
        x = np.loadtxt(files_1[i])
        y = np.loadtxt(files_2[i])
        z = np.loadtxt(files_3[i])
        t = np.loadtxt(files_4[i])
        n = files_1[i][-6:-4]  # 计数
        re = []
        for j in range(np.shape(x)[0]):
            re.append([x[j][0], x[j][1], y[j][1]])
            re.append([x[j][0], x[j][1], z[j][1]])
            re.append([x[j][0], x[j][1], t[j][1]])

            re.append([x[j][0], y[j][1], z[j][1]])
            re.append([x[j][0], y[j][1], t[j][1]])

            re.append([x[j][0], z[j][1], t[j][1]])

            if x[j][0] < 143:
                if files_1[i][-7:-6] == "l":  # 判断进站出站数据
                    if x[j][0] != x[j + 1][0]:
                        np.savetxt(
                            "data/train/data/station_" + n + "_arrival" + "_time_" + str(int(x[j][0])) + ".txt",
                            re,
                            fmt="%d")
                        re = []
                else:
                    if x[j][0] != x[j + 1][0]:
                        np.savetxt(
                            "data/train/data/station_" + n + "_departure" + "_time_" + str(
                                int(x[j][0])) + ".txt",
                            re,
                            fmt="%d")
                        re = []
            else:
                if files_1[i][-7:-6] == "l":
                    np.savetxt("data/train/data/station_" + n + "_arrival" + "_time_" + str(int(x[j][0])) + ".txt",
                               re, fmt="%d")
                else:
                    np.savetxt(
                        "data/train/data/station_" + n + "_departure" + "_time_" + str(int(x[j][0])) + ".txt",
                        re, fmt="%d")


# read_data(path='data/1')

# train_expand_zero(path='data/1')

# train_expend_time('data/new_1')

# train_expend_Y()

print('%f' % (time.time() - startTime) + "s")
