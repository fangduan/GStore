import os
import numpy as np
import pandas as pd
from attrdict import AttrDict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
gc.enable()

import preprocess
from models import LightGBM
import utils

train = pd.read_csv("~/Documents/Database/GStore/train_flatten.csv",dtype={'fullVisitorId': np.str})
test = pd.read_csv("~/Documents/Database/GStore/test_flatten.csv",dtype={'fullVisitorId': np.str})

## Preprocessing
# remove constant features
train = preprocess.rmConstant(train)
test = preprocess.rmConstant(test)
# Identify numeric & categorical features
num_cols, cat_cols = preprocess.col_dtypes(train)

# differentiate ID/time/target columns
ID_cols = ['fullVisitorId','sessionId','visitId']
time_cols = ['date','visitStartTime']
target_cols = ['totals_transactionRevenue']

num_cols = [c for c in num_cols if c not in ID_cols + time_cols + target_cols]
cat_cols = [c for c in cat_cols if c not in ID_cols]
num_cols.remove('trafficSource_adwordsClickInfo.page')
cat_cols.append('trafficSource_adwordsClickInfo.page')
cat_cols.remove('trafficSource_campaignCode')

for c in cat_cols:
    le = LabelEncoder()
    train_vals = list(train[c].values.astype(str))
    test_vals = list(test[c].values.astype(str))
    le.fit(train_vals + test_vals)
    train[c] = le.transform(train_vals)
    test[c] = le.transform(test_vals)

train[target_cols] = train[target_cols].fillna(0).astype('float')
train_idx = train['fullVisitorId']
test_idx = test['fullVisitorId']

## Modeling
TARGET = np.log1p(train['totals_transactionRevenue'].values)

train_X = train[num_cols + cat_cols + time_cols].copy()
test_X = test[num_cols + cat_cols + time_cols].copy()

# LightGBM
lgb_params = {"objective" : "regression",
              "boosting_type" : "dart",
              "metric" : "rmse",
              "num_leaves" : 15,
              "learning_rate" : 0.1,
              "max_depth" : 7,
              "bagging_fraction" : 0.9,
              "feature_fraction" : 0.9,
              "number_boosting_rounds" : 100,
              "early_stopping_rounds" : 10}
train_x, valid_x, train_y, valid_y = train_test_split(train_X, TARGET, test_size = 0.3, random_state = 1)
light_gbm = LightGBM(**lgb_params)
light_gbm.fit(train_x, train_y, valid_x, valid_y)
gc.collect()
pred = light_gbm.transform(test_X)

submission = utils.create_submission(pred, test)
submission.to_csv("first_trial.csv", index=False)
