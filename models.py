from attrdict import AttrDict

import numpy as np
import pandas as pd
import lightgbm as lgb

class LightGBM:
    def __init__(self, name = None, **params):
        self.params = params
        self.training_params = ['number_boosting_rounds','early_stopping_rounds']

    @property
    def model_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param not in self.training_params})

    @property
    def training_config(self):
        return AttrDict({param: value for param, value in self.params.items()
                         if param in self.training_params})

    def fit(self, X, y, X_valid, y_valid, **kwargs):
        data_train = lgb.Dataset(data = X,
                                 label = y,
                                 **kwargs)
        data_valid = lgb.Dataset(data = X_valid,
                                 label = y_valid,
                                 **kwargs)
        self.estimator = lgb.train(self.model_config,
                                   data_train,
                                   valid_sets = data_valid,
                                   num_boost_round = self.training_config.number_boosting_rounds,
                                   early_stopping_rounds = self.training_config.early_stopping_rounds,
                                   **kwargs)
        return self

    def transform(self, X, **kwargs):
        prediction = self.estimator.predict(X, num_iteration = self.estimator.best_iteration)
        return {'prediction': prediction}
