
import pandas as pd
import numpy as np

from Utils import *

train_csv = load_train()
for coupling_type in Config.MODEL_PARAMS.keys():
    n_atoms = Config.MODEL_PARAMS[coupling_type]
    XY_data = build_XY(train_csv, coupling_type, n_atoms, False)
    XY_data.to_csv(f'{Config.INPUT_XY}/{coupling_type}.csv')

#save_XY_Data()

