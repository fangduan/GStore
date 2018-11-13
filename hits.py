# https://www.kaggle.com/usmanabbas/flatten-hits-and-customdimensions
import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
from ast import literal_eval
import preprocess

MAXROWS = 10000
json_columns = ['device', 'geoNetwork', 'totals', 'trafficSource']

df = pd.read_csv('/Users/fangduan/Documents/Database/GStore/train_v2.csv',
                converters = {column: json.loads for column in json_columns},
                nrows = MAXROWS,
                dtype={'fullVisitorId': 'str'})
df.head()

df['hits']=df['hits'].apply(literal_eval)
df['hits']=df['hits'].str[0]
df=df[pd.notnull(df['hits'])]

df['customDimensions']=df['customDimensions'].apply(literal_eval)
df['customDimensions']=df['customDimensions'].str[0]
df=df[pd.notnull(df['customDimensions'])]

json_columns = ['device', 'geoNetwork', 'totals', 'trafficSource','hits','customDimensions']

for column in json_columns:
    column_as_df = json_normalize(df[column])
    column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
    df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

(df != df.iloc[0]).any()
