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
    'boosting_type': 'gbdt'
}

MODEL_PARAMS = {
        '1JHN': 11,
        '1JHC': 11,
        '2JHH': 11,
        '2JHN': 11,
        '2JHC': 11,
        '3JHH': 11,
        '3JHC': 11,
        '3JHN': 11
    }

COUNT_DISTANCES = 6