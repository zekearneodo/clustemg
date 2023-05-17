import logging

import itertools
import umap
import hdbscan
import pandas as pd
import numpy as np

from sklearn import metrics
from scipy import stats
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm

logger = logging.getLogger("clustemg.umapreduce")

def flatten_spectrograms(specs: np.array) -> np.array:
    # flatten all the spectrograms for input to umap

    return np.reshape(specs, (np.shape(specs)[0], np.prod(np.shape(specs)[1:])))

def embed_cluster_umap(df: pd.DataFrame, min_dist, n_neighbors, min_samples,
                       key: str='emg_pad', verbose=True) -> pd.DataFrame:

  # reduce using umap
  features_flattened = flatten_spectrograms(list(df[key]))
  umap_reducer = umap.UMAP(min_dist = min_dist, n_neighbors=n_neighbors, 
                           verbose = verbose,
                           random_state=42)
  
  umap_key = 'umap-{}-{}-{}'.format(key, min_dist, n_neighbors)
  logger.info('reducing using umap to {}'.format(umap_key))
  umap_embedding = umap_reducer.fit_transform(features_flattened)
  df[umap_key] = list(umap_embedding)

  # cluster using hdbscan
  clu_key = 'clu_hdbs-{}-{}'.format(umap_key, min_samples)
  logger.info('clustering using hdbscan to column {}'.format(clu_key))

  umap_labels = hdbscan.HDBSCAN(
      min_samples=min_samples,
      min_cluster_size=2,
  ).fit_predict(umap_embedding)
  clu_key = 'clu_hdbs-{}-{}'.format(umap_key, min_samples)
  df[clu_key] = list(umap_labels)

  

  if verbose:
    score = metrics.silhouette_score(umap_embedding, umap_labels)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=umap_labels, cmap='Spectral', s=5)
    ax.set_title('Umap/hdbscan {}, score {}'.format(clu_key, score))
    logger.info('{}, {} found {} clusters'.format(umap_key, clu_key, np.unique(umap_labels)))

  return df, umap_key, clu_key


def clu_parameter_sweep(par_dict: dict, t_df: pd.DataFrame, 
                    emg_key: str='emg_pad', 
                    verbose: bool=False):
  # make a sweep across umap parameters and check for optimal
  # clusterings

  umap_clu_keys_list = []
  umap_keys_list = []

  d_list = par_dict['min_dist_list']
  n_list = par_dict['n_neighbor_list']
  s_list = par_dict['min_samples_list']

  iterations = list(itertools.product(*[d_list, n_list, s_list]))

  for d, n, s in tqdm(iterations, total=len(iterations)):
  #for d, n, s in itertools.product(*[[0.01], [5, 10], [5]]):
    t_df, umap_key, clu_key = embed_cluster_umap(t_df, d, n, s, 
                                                 key=emg_key,
                                                 verbose=verbose)
    umap_clu_keys_list.append(clu_key)
    umap_keys_list.append(umap_key)

  # make dataframe of clustering/projection meta/score
  clu_meta = pd.DataFrame({'clu_key': umap_clu_keys_list,
                         'umap_key': umap_keys_list, 
                        'n_clu': [np.unique(t_df.loc[t_df[x]>=0, x]).size for x in umap_clu_keys_list],
                        'n_out': [np.sum(t_df[x]<0) for x in umap_clu_keys_list],
                         'score': [metrics.silhouette_score(np.vstack(t_df[x]), t_df[y]) for x, y in zip(umap_keys_list, umap_clu_keys_list)]

                         })

  clu_meta['umap-min_dist'], clu_meta['umap-n_neighbors'], clu_meta['hdbs-min_samples'] = zip(*clu_meta['clu_key'].apply(lambda s: s.split('-')[-3:]))

  clu_meta.sort_values(['score', 'n_clu'], ascending=False, inplace=True)
  clu_meta.reset_index(inplace=True)
  
  
  n_clu = stats.mode(clu_meta['n_clu'], keepdims=False).mode
  best_clu_idx = clu_meta[(clu_meta['n_clu']==n_clu)].index[0]
  logger.info('Best clustering is {} clusters, with parameters {}'.format(n_clu, 
              clu_meta.iloc[best_clu_idx]))
  
  ### plot the best clustering
  clu_meta_s = clu_meta.iloc[best_clu_idx]
  emb = np.stack(t_df[clu_meta_s['umap_key']])
  labels = t_df[clu_meta_s['clu_key']]
  score = metrics.silhouette_score(emb, labels)
  #print(score)
  fig, ax = plt.subplots(nrows=1, ncols=1)
  ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap='Set1', s=3, marker='o')
  ax.set_title('Best cluster {}, score {}'.format(clu_key, score))
  
  return t_df, clu_meta, best_clu_idx
