import numpy as np

def rmConstant(data):
    data = data.loc[:, (data != data.iloc[0]).any()]
    return(data)

def col_dtypes(data):
    num_cols = data.select_dtypes(include = [np.number]).columns
    cat_cols = data.select_dtypes(include = [np.object]).columns
    return num_cols, cat_cols
