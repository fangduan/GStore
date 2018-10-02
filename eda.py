import os
import numpy as np
import pandas as pd
import gc
gc.enable()

train = pd.read_csv("/Users/fangduan/Documents/Database/GStore/train_flatten.csv")
train.shape
train_numeric_feature = train.select_dtypes(include = [np.number])
train_categorical_feature = train.select_dtypes(include = [np.object])
# remove constant features
train = train.loc[:, (train != train.iloc[0]).any()]
train.shape
train_categorical_feature.isnull().sum().sort_values(ascending = False)
train_numeric_feature.isnull().sum().sort_values(ascending = False)
