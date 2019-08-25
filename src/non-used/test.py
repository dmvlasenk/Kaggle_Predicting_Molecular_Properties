from Utils import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import os.path





coupling_type = '1JHC'
XY_Data = pd.read_csv(f'{Config.INPUT_XY}/{coupling_type}.csv', index_col=0)

train_csv = load_train()
cur_train_csv = train_csv[train_csv.type == coupling_type]


group_train = cur_train_csv.molecule_index.values


aa = 1