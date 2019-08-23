# -*- coding: utf-8 -*-
# config file

DATA_PATH = '..\\input'
INPUT_ADDED = '..\\input_added'
INPUT_XY = '..\\input_added\\XY'
INPUT_XY_FILTER = '..\\input_added\\XY_filter'
SUBMISSIONS_PATH = '..\\output'

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 50,
    'min_child_samples': 30,
    'max_depth': 11,
    'reg_alpha': 0.01,
    'reg_lambda': 0.3,
    'bagging_freq': 2000,
    'bagging_fraction': 0.7,
    'bagging_seed': 11,
    'colsample_bytree': 1.0
}

MODEL_PARAMS = {
        '1JHN': 10,
        '1JHC': 15,
        '2JHH': 13,
        '2JHN': 13,
        '2JHC': 13,
        '3JHH': 13,
        '3JHC': 15,
        '3JHN': 15
    }