#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Utils import *
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
import matplotlib as plt

plt.interactive(False)


from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


def trainModel(X_data, y_data):
    X_train = X_data
    X_val = X_data
    y_train = y_data
    y_val = y_data
    categorical_features = [col for col in X_train if col.startswith('atom_')]
    # X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.3, random_state=228)
    # to record eval results for plotting
    model = LGBMRegressor(**Config.LGB_PARAMS, n_estimators=500, n_jobs=-1)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae',
              verbose=100, early_stopping_rounds=500,
              categorical_feature=categorical_features)
    return model

def get_oultileners_error(X_data, y_data, y_n_pred):
    X_non_filter = X_data
    y_non_filter = y_data
    err = abs(y_non_filter - y_n_pred)
    _, up_err = np.percentile(err, [0, 99.5])
    dp = y_non_filter.to_frame()
    dp['err'] = err #pd.Series(err)
    dpNewErr = dp[dp.err >= up_err]
    return  dpNewErr

def rewrite_filter_file(coupling_type, XY_Data):
    n_splits = 9
    n_folds = n_splits
    dpOldErr = get_filtered_errors(coupling_type)
    XY_Filtered = get_filtered_xy(XY_Data, dpOldErr)
    X_data, y_data = build_x_y_data(XY_Filtered)

    columns = X_data.columns
    random_state = 128
    y_pred = y_data * 0
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_index, val_index) in enumerate(kfold.split(X_data, y_data)):
        X_train, X_val = X_data[columns].iloc[train_index], X_data[columns].iloc[val_index]
        y_train, y_val = y_data.iloc[train_index], y_data.iloc[val_index]
        model = trainModel(X_train, y_train)
        y_fold_pred = model.predict(X_val)
        y_pred.iloc[val_index] += y_fold_pred
    dpNewErr = get_oultileners_error(X_data, y_data, y_pred)
    plt.plot(y_data, y_pred)


    if dpOldErr is None:
        dfErr = dpNewErr
    else:
        dfErr = pd.concat([dpOldErr, dpNewErr])
    dfErr.to_csv(f'{Config.INPUT_XY_FILTER}/{coupling_type}.csv')

def writeFilter(coupling_type):
    XY_Data1 = pd.read_csv(f'{Config.INPUT_XY}/{coupling_type}.csv', index_col=0)
    for iter in range(0, filter_count):
        rewrite_filter_file(coupling_type, XY_Data1)


coupling_type = '1JHN'
filter_count = 1
writeFilter(coupling_type)
#for coupling_type in Config.MODEL_PARAMS.keys():
#    writeFilter(coupling_type)
