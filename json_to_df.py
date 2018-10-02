import os
import json
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize

print(os.listdir("../Documents/Database/GStore"))
json_columns = ['device', 'geoNetwork','totals', 'trafficSource']
filepath = "../Documents/Database/GStore/"
def load_df(filepath, filename):
    csv_path = filepath + filename
    df = pd.read_csv(csv_path,
                     converters={column: json.loads for column in json_columns},
                     dtype={'fullVisitorId': 'str'})

    for column in json_columns:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df
train = load_df(filepath, "train.csv")
test = load_df(filepath,"test.csv")
train.to_csv(filepath + "train_flatten.csv", index = False)
test.to_csv(filepath + "test_flatten.csv", index = False)
