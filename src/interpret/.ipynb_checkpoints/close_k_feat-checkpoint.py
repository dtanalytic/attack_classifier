import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def compute_all_distances(feat_data, thresh_rough, batch_size=1000):
    
    n_samples = feat_data.shape[0]
    indices = np.arange(n_samples)
    results = []
    # import pdb;pdb.set_trace()
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        # Вычисляем попарные расстояния для батча
        dist_batch = cosine_distances(feat_data.iloc[batch_indices], feat_data)
        
        # Фильтруем с грубым фильтром, чтобы осталось что-то, потом уже выберем, сколько надо нам
        dist_df = pd.DataFrame(dist_batch, index=batch_indices).stack()
        thresh = dist_df.loc[lambda x: x!=0].quantile(thresh_rough)
        dist_df = dist_df.loc[lambda x: (x>0)&(x < thresh)]
        results.append(dist_df)
        select_df = pd.concat(results)
        
    return select_df

def compute_selected_distances(feat_data, idx):
    
    dist_df = pd.DataFrame(cosine_distances(feat_data.iloc[idx], feat_data), index=idx)
    select_df = dist_df.stack().loc[lambda x: x!=0]

    return select_df

def select_k_neigh(data, feat_data, k, idx=None, **kwargs):

    if not idx:
        select_df = compute_all_distances(feat_data, **kwargs)
    else:
        select_df = compute_selected_distances(feat_data, idx)

    select_df = select_df.reset_index().rename(columns={0:'val', 'level_0':'mem1', 'level_1':'mem2'})\
         .groupby('mem1').apply(lambda x: x.sort_values(by='val', ascending=True).iloc[:k]).reset_index(drop=True)
    
    pairs = select_df[['mem1', 'mem2']].apply(lambda x: tuple(x), axis=1).tolist()
    # pairs = [(it1, it2) for it1, it2 in pairs if it1<it2 ]
    comp_df = data.loc[data.index[[it1 for it1, _ in pairs]], ['sentence', 'labels', 'url']].reset_index(drop=True).join(
    data.loc[data.index[[it2 for _, it2 in pairs]], ['sentence', 'labels', 'url']].reset_index(drop=True), 
    lsuffix='_left', rsuffix='_right'
    )
     
    
    return comp_df, select_df