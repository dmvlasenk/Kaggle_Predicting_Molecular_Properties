import pandas as pd
import numpy as np


coupling_type = '1JHN'


INPUT_ADDED = '..\\input_added'
DATA_PATH = '..\\input'


brute_columns = [
    'molecule_atom_index_0_dist_min',
    'molecule_atom_index_0_dist_max',
    'molecule_atom_index_1_dist_min',
    'molecule_atom_index_0_dist_mean',
    'molecule_atom_index_0_dist_std',
    'dist',
    'molecule_atom_index_1_dist_std',
    'molecule_atom_index_1_dist_max',
    'molecule_atom_index_1_dist_mean',
    'molecule_atom_index_0_dist_max_diff',
    'molecule_atom_index_0_dist_max_div',
    'molecule_atom_index_0_dist_std_diff',
    'molecule_atom_index_0_dist_std_div',
    'atom_0_couples_count',
    'molecule_atom_index_0_dist_min_div',
    'molecule_atom_index_1_dist_std_diff',
    'molecule_atom_index_0_dist_mean_div',
    'atom_1_couples_count',
    'molecule_atom_index_0_dist_mean_diff',
    'molecule_couples',
    'molecule_dist_mean',
    'molecule_atom_index_1_dist_max_diff',
    'molecule_atom_index_0_y_1_std',
    'molecule_atom_index_1_dist_mean_diff',
    'molecule_atom_index_1_dist_std_div',
    'molecule_atom_index_1_dist_mean_div',
    'molecule_atom_index_1_dist_min_diff',
    'molecule_atom_index_1_dist_min_div',
    'molecule_atom_index_1_dist_max_div',
    'molecule_atom_index_0_z_1_std',
    'molecule_type_dist_std_diff',
    'molecule_atom_1_dist_min_diff',
    'molecule_atom_index_0_x_1_std',
    'molecule_dist_min',
    'molecule_atom_index_0_dist_min_diff',
    'molecule_atom_index_0_y_1_mean_diff',
    'molecule_type_dist_min',
    'molecule_atom_1_dist_min_div',
    'molecule_dist_max',
    'molecule_atom_1_dist_std_diff',
    'molecule_type_dist_max',
    'molecule_atom_index_0_y_1_max_diff',
    'molecule_type_0_dist_std_diff',
    'molecule_type_dist_mean_diff',
    'molecule_atom_1_dist_mean',
    'molecule_atom_index_0_y_1_mean_div',
    'molecule_type_dist_mean_div']
    
def take_n_atoms(df, brute_columns, n_atoms, four_start=4):
        

    labels = []
    for i in range(2, n_atoms):
        label = f'atom_{i}'
        labels.append(label)

    for i in range(1, n_atoms):
        labels.append(f'r_x_{i}')
    for i in range(2, n_atoms):
        labels.append(f'r_y_{i}')
    for i in range(3, n_atoms):
        labels.append(f'r_z_{i}')
    if 'scalar_coupling_constant' in df:
        labels.append('scalar_coupling_constant')
        
    labels = labels + brute_columns        
    output = df[labels]
    #atoms_names = list([col for col in output if col.startswith('atom_')])[2:]
    #output = output.drop(atoms_names, axis=1)
    return output

if False:
    train_dtypes = {
        'molecule_name': 'category',
        'atom_index_0': 'int8',
        'atom_index_1': 'int8',
        'type': 'category',
        'scalar_coupling_constant': 'float32'
    }
    train_csv = pd.read_csv(f'{DATA_PATH}\\train.csv', index_col='id', dtype=train_dtypes)
    train_csv['molecule_index'] = train_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
    train_csv = pd.read_csv(f'{DATA_PATH}\\train.csv', index_col='id', dtype=train_dtypes)
    cur_train = train_csv[train_csv.type == coupling_type]


full_all =  pd.read_csv(f'{INPUT_ADDED}/{coupling_type}.csv')
full = full_all 
n_atoms = 10


brute_all =  pd.read_csv(f'{INPUT_ADDED}/brute_X_{coupling_type}.csv')
full = full.merge(brute_all, left_on='id', right_on='Unnamed: 0')

#cur_x_test.to_csv(f'{INPUT_ADDED}/brute_X_test_{coupling_type}.csv')    

df = take_n_atoms(full, brute_columns, n_atoms)
df.describe()

