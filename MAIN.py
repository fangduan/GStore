import os
import numpy as np
import pandas as pd
from attrdict import AttrDict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import lightgbm as lgb

import preprocess
from models import LightGBM
import utils

import gc
gc.enable()

train = pd.read_csv("~/Documents/Database/GStore/train_flatten.csv",dtype={'fullVisitorId': np.str})
test = pd.read_csv("~/Documents/Database/GStore/test_flatten.csv",dtype={'fullVisitorId': np.str})
train.shape, test.shape

random_seed = 1024
########################## Preprocessing #######################################
#### remove constant features
train = preprocess.rmConstant(train)
test = preprocess.rmConstant(test)
train.shape, test.shape
set(train.columns) - set(test.columns)
train = train.drop(columns = 'trafficSource_campaignCode') # this column only exists in train set

# Identify numeric & categorical features
num_cols, cat_cols = preprocess.col_dtypes(train)

#### differentiate ID/time/target columns
ID_cols = ['fullVisitorId','sessionId','visitId']
time_cols = ['date','visitStartTime']
target_cols = ['totals_transactionRevenue']

num_cols = [c for c in num_cols if c not in ID_cols + time_cols + target_cols]
cat_cols = [c for c in cat_cols if c not in ID_cols]
num_cols.remove('trafficSource_adwordsClickInfo.page')
cat_cols.append('trafficSource_adwordsClickInfo.page')

#### fill NAs
# drop features with na_percentage > 90%, add indicator_NA feature later
drop_cols = ['trafficSource_adContent',
             'trafficSource_adwordsClickInfo.slot',
             'trafficSource_adwordsClickInfo.adNetworkType',
             'trafficSource_adwordsClickInfo.isVideoAd',
             'trafficSource_adwordsClickInfo.page',
             'trafficSource_adwordsClickInfo.gclId']

for df in [train, test]:
    df['trafficSource_isTrueDirect'].fillna('missing', inplace = True)
    df['trafficSource_referralPath'].fillna('missing', inplace = True)
    df['trafficSource_keyword'].fillna('missing', inplace = True)
    df['totals_bounces'].fillna(0.0, inplace = True)
    df['totals_newVisits'].fillna(0.0, inplace = True)
    df['totals_pageviews'].fillna(0.0, inplace = True)
del df
train = train.drop(columns = drop_cols)
test = test.drop(columns = drop_cols)

cat_cols = [c for c in cat_cols if c not in drop_cols]
# encoder
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
train_X = train[num_cols + cat_cols].copy()
test_X = test[num_cols + cat_cols].copy()

folds = utils.get_folds(train, n_splits = 5)

oof_prediction = np.zeros(train_X.shape[0])
sub_prediction = np.zeros(test_X.shape[0])
oof_scores = []

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

light_gbm = LightGBM(**lgb_params)

for fold_, (trn_, val_) in enumerate(folds):
    trn_X, trn_y = train_X.iloc[trn_], TARGET[trn_]
    val_X, val_y = train_X.iloc[val_], TARGET[val_]

    # la.fit(trn_X, trn_y)
    light_gbm.fit(trn_X, trn_y, val_X, val_y)
    # oof_prediction[val_] = la.predict(val_X)
    oof_prediction[val_] = light_gbm.transform(val_X)['prediction']
    oof_prediction[oof_prediction < 0] = 0
    # _preds = la.predict(test_X)
    _preds = light_gbm.transform(test_X)['prediction']
    _preds[_preds < 0 ] = 0
    sub_prediction += np.expm1(_preds) / len(folds)
    oof_scores.append(mean_squared_error(TARGET[val_], oof_prediction[val_])**0.5)
    print('Fold %d RMSE : %.5f' % (fold_ + 1, oof_scores[-1]))
    gc.collect()

# Lasso
la = linear_model.Lasso()
TARGET = pd.DataFrame(TARGET)
for fold_, (trn_, val_) in enumerate(folds):
    trn_X, trn_y = train_X.iloc[trn_], TARGET.iloc[trn_]
    val_X, val_y = train_X.iloc[val_], TARGET.iloc[val_]

    la.fit(trn_X, trn_y)
    oof_prediction[val_] = la.predict(val_X)
    oof_prediction[oof_prediction < 0] = 0
    _preds = la.predict(test_X)
    _preds[_preds < 0 ] = 0
    sub_prediction += np.expm1(_preds) / len(folds)
    oof_scores.append(mean_squared_error(TARGET.iloc[val_], oof_prediction[val_])**0.5)
    print('Fold %d RMSE : %.5f' % (fold_ + 1, oof_scores[-1]))
    gc.collect()

submission = utils.create_submission(sub_prediction, test)
submission.to_csv("first_trial.csv", index=False)
