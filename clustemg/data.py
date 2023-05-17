import os
import logging

import h5py
import json
import pandas as pd
import numpy as np

from scipy import stats
from matplotlib import pyplot as plt

logger = logging.getLogger("clustemg.data")

def load_raw_data(data_folder: str) -> tuple:
    # load all the trials into a two column pandas dataframe (each row is a word)
    # in the 'emg' key, a npy array with [i_arr, x, y, t] has the emg sampled at 1khz
    h5_path = os.path.join(data_folder, 'raw_samples.h5')
    json_path = os.path.join(data_folder, 'session_info.json')
    
    logger.info('loading the data from {}, {} into a pandas DataFrame'.format(json_path, h5_path))
    
    with open(json_path, 'r') as f:
        meta_dict = json.load(f)
    
    all_trial = []
    all_emg = []
    with h5py.File(h5_path, 'r') as f:
        for trial_idx in f.keys():
            trial_grp = f[trial_idx]
            all_trial.append(trial_idx)
            all_emg.append(trial_grp['emg'][:])
    
    trial_df = pd.DataFrame({'trial_idx': all_trial,
                 'emg': all_emg})
    trial_df['word_len'] = trial_df['emg'].apply(lambda x: x.shape[-1])

    logger.info('rectifying and z_scoring all trials')
    trial_df['emg_z'] = trial_df['emg'].apply(lambda x: stats.zscore(x, axis=-1))
    trial_df['emg_abs'] = trial_df['emg_z'].apply(lambda x: np.abs(x))
    return trial_df, meta_dict


def histogram_with_bins(data, num_bins, plot=True, ax=None) -> tuple:
    # Create histogram and get bin edges
    hist, bin_edges = np.histogram(data, bins=num_bins)

    # Get indices of bins for each data point
    bin_indices = np.digitize(data, bin_edges)

    # Plot histogram
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(data, bins=num_bins)
        ax.set_xlabel('T (ms)')
        ax.set_title('Distribution of word durations T')

    # Return histogram and bin indices
    return hist, bin_indices