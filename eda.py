import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
gc.enable()


train = pd.read_csv("/Users/fangduan/Documents/Database/GStore/train_flatten.csv",dtype={'fullVisitorId': np.str})
test = pd.read_csv("/Users/fangduan/Documents/Database/GStore/test_flatten.csv",dtype={'fullVisitorId': np.str})
train.shape

# remove constant features
train = train.loc[:, (train != train.iloc[0]).any()]
test = test.loc[:, (test != test.iloc[0]).any()]
train.shape
test.shape

# Identify numeric & categorical features
num_cols = train.select_dtypes(include = [np.number]).columns
cat_cols = train.select_dtypes(include = [np.object]).columns
# calculate NA
train.select_dtypes(include = [np.object]).isnull().sum().sort_values(ascending = False)
train.select_dtypes(include = [np.number]).isnull().sum().sort_values(ascending = False)

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
              "feature_fraction" : 0.9}

train_x, valid_x, train_y, valid_y = train_test_split(train_X, TARGET, test_size = 0.3, random_state = 1)
lgb_train = lgb.Dataset(train_x, label = train_y)
lgb_val = lgb.Dataset(valid_x, label = valid_y)
model = lgb.train(lgb_params,
                  lgb_train,
                  valid_sets = lgb_val,
                  num_boost_round = 1000,
                  early_stopping_rounds = 100,
                  verbose_eval=20)

pred = model.predict(test_X, num_iteration = model.best_iteration)
test["PredictedLogRevenue"] = np.expm1(pred)
submission = test.groupby("fullVisitorId").agg({"PredictedLogRevenue" : "sum"}).reset_index()
submission["PredictedLogRevenue"] = np.log1p(submission["PredictedLogRevenue"])
submission["PredictedLogRevenue"] =  submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission['PredictedLogRevenue'] = submission['PredictedLogRevenue'].fillna(0).astype('float')
submission.to_csv("first_trial.csv", index=False)
submission
