from setting import *
import torch
import numpy as np
import pandas as pd
from torch.utils import data
train_data = pd.read_csv(train_url)
test_data = pd.read_csv(test_url)
# print(train_data.shape)
# print(test_data.shape)
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))


# 1. 标准化处理
def standardize():
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)


# 非标准化处理
def nonStandardize():
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    for feature, data in all_features[numeric_features].iteritems():
        data = data.fillna(data.mean())


# 2. 将非数值特征转化为独热编码
def oneHot():
    global all_features
    all_features = pd.get_dummies(all_features, dummy_na=True)


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


standardize()
oneHot()
n_train = train_data.shape[0]
train_features = torch.tensor(all_features.iloc[0:n_train, :].values, dtype=torch.float32)
test_features = torch.tensor(all_features.iloc[n_train:, :].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.iloc[:, -1].values.reshape(-1, 1), dtype=torch.float32)
dataLoader = load_array((train_features, train_labels), batch_size)

