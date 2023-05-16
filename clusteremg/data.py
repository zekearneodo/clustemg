import os
import logging

import h5py
import pandas as pd

from scipy import stats

logger = logging.getLogger("clustemg.data")

def load_raw_data(data_folder: str) -> pd.DataFrame:
    # load all the trials into a two column pandas dataframe (each row is a word)
    # in the 'emg' key, a npy array with [i_arr, x, y, t] has the emg sampled at 1khz
    h5_path = os.path.join(data_folder, 'raw_samples.h5')
    json_path = os.path.join(data_folder, 'session_info.json')
    logger.info('loading the data from {}, {} into a pandas DataFrame'.format(json_path, h5_path))

    all_trial = []
    all_emg = []
    with h5py.File(h5_path, 'r') as f:
        for trial_idx in f.keys():
            trial_grp = f[trial_idx]
            all_trial.append(trial_idx)
            all_emg.append(trial_grp['emg'][:])
            
    trial_df = pd.DataFrame({'trial_idx': all_trial,
                 'emg': all_emg})
    
    trial_df['emg_z'] = trial_df['emg'].apply(lambda x: stats.zscore(x, axis=-1))
    trial_df['emg_abs'] = trial_df['emg_z'].apply(lambda x: np.abs(x))
    
    trial_df['word_len'] = trial_df['emg'].apply(lambda x: x.shape[-1])
    return trial_df