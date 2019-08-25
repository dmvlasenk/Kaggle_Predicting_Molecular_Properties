import pandas as pd
import numpy as np
import Config
import os.path


def get_filtered_errors(coupling_type):
    filterFile = f'{Config.INPUT_XY_FILTER}/{coupling_type}.csv'
    isFilter = os.path.exists(filterFile)
    if  not isFilter:
        dpOldErr = None
    else:
        dpOldErr = pd.read_csv(f'{Config.INPUT_XY_FILTER}/{coupling_type}.csv', index_col=0)
    return dpOldErr

def get_filtered_xy(XY_Data, dpOldErr):
    if dpOldErr is None:
        XY_Filtered = XY_Data
    else:
        mask = XY_Data.index.isin(dpOldErr.index)
        mask = np.invert(mask)
        XY_Filtered = XY_Data.loc[mask]
        XY_Filtered.describe()
    return XY_Filtered

def build_x_y_data(df):
    if 'scalar_coupling_constant' in df:
        X_data = df.drop(['scalar_coupling_constant'], axis=1)
        y_data = df['scalar_coupling_constant']
    else:
        X_data = df
        y_data = None
    return X_data, y_data

def add_distance_between(df, suffix1, suffix2):
    df[f'd_{suffix1}_{suffix2}'] = ((
        (df[f'x_{suffix1}'] - df[f'x_{suffix2}'])**np.float32(2) +
        (df[f'y_{suffix1}'] - df[f'y_{suffix2}'])**np.float32(2) +
        (df[f'z_{suffix1}'] - df[f'z_{suffix2}'])**np.float32(2)
    )**np.float32(0.5))

def add_distances(df, n_atoms):
    for i in range(1, n_atoms):
        for vi in range(min(Config.COUNT_DISTANCES, i)):
            add_distance_between(df, i, vi)

def take_n_atoms(df, n_atoms, four_start=Config.COUNT_DISTANCES):
    labels = []
    for i in range(2, n_atoms):
        label = f'atom_{i}'
        labels.append(label)
    if True:
        for i in range(1, n_atoms):
            labels.append(f'r_x_{i}')
        for i in range(2, n_atoms):
            labels.append(f'r_y_{i}')
        for i in range(3, n_atoms):
            labels.append(f'r_z_{i}')
    if True:
        for i in range(2, n_atoms):
            num = min(i, Config.COUNT_DISTANCES) if i < four_start else Config.COUNT_DISTANCES
            for j in range(num):
                labels.append(f'd_{i}_{j}')
        #labels.remove('d_1_0')
    #dependent_vals = ['d_2_0', 'd_2_1', 'd_3_0', 'd_9_1']
    #labels = [x for x in labels if x not in dependent_vals]



    if 'scalar_coupling_constant' in df:
        labels.append('scalar_coupling_constant')
    #labels = labels + brute_columns
    output = df[labels]
    #atoms_names = list([col for col in output if col.startswith('atom_')])[2:]
    #output = output.drop(atoms_names, axis=1)
    return output

def build_XY(some_csv, coupling_type, n_atoms, isTest = False):
    #full = build_couple_dataframe(some_csv, structures_csv, coupling_type, n_atoms=n_atoms)
    if isTest:
        full_all = pd.read_csv(f'{Config.INPUT_ADDED}/test_{coupling_type}.csv')
    else:
        full_all =  pd.read_csv(f'{Config.INPUT_ADDED}/{coupling_type}.csv')
    add_distances(full_all,n_atoms)
    #brute_all =  pd.read_csv(f'{INPUT_ADDED}/brute_X_{coupling_type}.csv')
    #full =  full_all.merge(brute_all, left_on='id', right_on='Unnamed: 0')
    full = full_all

    df = take_n_atoms(full, n_atoms)
    df = df.fillna(0)
    #print(df.columns
    return df

train_dtypes = {
    'molecule_name': 'category',
    'atom_index_0': 'int8',
    'atom_index_1': 'int8',
    'type': 'category',
    'scalar_coupling_constant': 'float32'
}


def load_train():
    train_csv = pd.read_csv(f'{Config.DATA_PATH}\\train.csv', index_col='id', dtype=train_dtypes)
    train_csv['molecule_index'] = train_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
    train_csv = train_csv[['molecule_index', 'atom_index_0', 'atom_index_1', 'type', 'scalar_coupling_constant']]
    return  train_csv

