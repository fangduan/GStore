import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import gc
gc.enable()
import preprocess

train = pd.read_csv("~/Documents/Database/GStore/train_flatten.csv",dtype={'fullVisitorId': np.str})
test = pd.read_csv("~/Documents/Database/GStore/test_flatten.csv",dtype={'fullVisitorId': np.str})

########################## Preprocessing #######################################
#### remove constant features
train = preprocess.rmConstant(train)
test = preprocess.rmConstant(test)
train.shape
test.shape
set(train.columns) - set(test.columns)
train = train.drop(columns = 'trafficSource_campaignCode')

# Identify numeric & categorical features
num_cols, cat_cols = preprocess.col_dtypes(train)

#### calculate NA
NA_percent = (train.isnull().sum()/train.isnull().count()).reset_index()
NA_percent.columns = ['features', 'percentage']
NA_percent = NA_percent.loc[NA_percent['percentage'] > 0]
NA_percent = NA_percent.sort_values(by = 'percentage', ascending = False)

# sample code for barplot of %MissingValue
idx = np.arange(NA_percent.shape[0])
fig, ax = plt.subplots(figsize = (20,5))
ax.barh(idx, NA_percent.percentage.values)
ax.set_yticks(idx)
ax.set_yticklabels(NA_percent.features.values,rotation = 'horizontal')
ax.set_xlabel('missing value percentage')
ax.set_title('Plot of missing value percentage')
plt.show()
# fig.savefig("./plot/plot_NA_perc.pdf") # save plot

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
#
for df in [train, test]:
    df['trafficSource_isTrueDirect'].fillna('missing', inplace = True)
    df['trafficSource_referralPath'].fillna('missing', inplace = True)
    df['trafficSource_keyword'].fillna('missing', inplace = True)
    df['totals_bounces'].fillna(0.0, inplace = True)
    df['totals_newVisits'].fillna(0.0, inplace = True)
    df['totals_pageviews'].fillna(0.0, inplace = True)

train = train.drop(columns = drop_cols)
test = test.drop(columns = drop_cols)

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
