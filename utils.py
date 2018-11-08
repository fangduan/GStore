import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

def get_folds(df = None, n_splits = 5):
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))
    folds = GroupKFold(n_splits = n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for train_vis, test_vis in folds.split(X = unique_vis, y = unique_vis, groups = unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[train_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[test_vis])]
            ]
        )
    return fold_ids

def create_submission(pred,test):

    pred = pd.DataFrame(pred)
    test["PredictedLogRevenue"] = pred
    submission = test.groupby("fullVisitorId").agg({"PredictedLogRevenue" : "sum"}).reset_index()
    submission["PredictedLogRevenue"] = np.log1p(submission["PredictedLogRevenue"])
    submission["PredictedLogRevenue"] =  submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
    submission['PredictedLogRevenue'] = submission['PredictedLogRevenue'].fillna(0).astype('float')
    return submission
