import numpy as np
import pandas as pd

def create_submission(pred,test):

    pred = pd.DataFrame(pred)
    test["PredictedLogRevenue"] = np.expm1(pred)
    submission = test.groupby("fullVisitorId").agg({"PredictedLogRevenue" : "sum"}).reset_index()
    submission["PredictedLogRevenue"] = np.log1p(submission["PredictedLogRevenue"])
    submission["PredictedLogRevenue"] =  submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
    submission['PredictedLogRevenue'] = submission['PredictedLogRevenue'].fillna(0).astype('float')
    return submission
