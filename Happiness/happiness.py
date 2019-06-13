import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('display.width', 100)


def load_train_data(train):
    # 删除训练集中无效的标签对应的数据
    train = train.loc[train['happiness'] != -8]

    # 删除无用的收入数据
    train = train.loc[train['family_income'] != -1]

    train = train.loc[train['family_income'] != -2]

    train = train.loc[train['family_income'] != -3]

    train = train.loc[train['family_income'] != -8]

    train = train.loc[train['family_income'] != 0]

    train = train.reset_index(drop=True)

    for i in range(len(train)):
        if np.isnan(train['family_income'][i]):
            train = train.drop(i)

    # 填写日期-生日 获得年龄
    train['Age'] = pd.to_datetime(train['survey_time']).dt.year - train['birth']

    # 筛选特征值
    features = train.corr()['happiness'][abs(train.corr()['happiness']) > 0.05]
    features = features.index.tolist()
    features.append("Age")
    train = train[features]
    train['family_income'] = train['family_income'] / 10000
    return train, features


def load_test_data(test, features):
    # 删除无用的收入数据
    test = test.loc[test['family_income'] != -1]

    test = test.loc[test['family_income'] != -2]

    test = test.loc[test['family_income'] != -3]

    test = test.loc[test['family_income'] != -8]

    test = test.loc[test['family_income'] != 0]

    test = test.reset_index(drop=True)

    for i in range(len(test)):
        if np.isnan(test['family_income'][i]):
            test = test.drop(i)

    # 填写日期-生日 获得年龄
    test['Age'] = pd.to_datetime(test['survey_time']).dt.year - test['birth']
    test = test[features]
    test['family_income'] = test['family_income'] / 10000
    return test
