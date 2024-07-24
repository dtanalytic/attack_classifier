import pandas as pd
import numpy as np
import joblib
import click
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer


from ruamel.yaml import YAML

import sys
sys.path.append('.')
from src.feat_gen import custom_tok


@click.command()
@click.argument('config_path')
def main(config_path):
    
    conf = YAML().load(open(config_path))
    np.random.seed(conf['seed'])
    
    mlb = joblib.load(conf['prep_text']['mlb_fn'])
    data = pd.read_csv(conf['feat_gen']['data_fn'])
    feat_data = joblib.load(conf['feat_gen']['feat_fn'])


    tr_idx = data.query('split=="tr"').index


    if conf['feat_eng']['tfidf_dim']:
        nmf = NMF(n_components=conf['feat_eng']['tfidf_dim'], random_state=conf['seed'])
        nmf.fit(feat_data.loc[tr_idx])
        feat_data_new = pd.DataFrame(nmf.transform(feat_data), index = feat_data.index, columns=nmf.get_feature_names_out())
    else:
        feat_data_new = feat_data

    if conf['feat_eng']['add_clust_feat']:
        k_l = np.arange(5,100,10)
        feat_clust_add = pd.DataFrame()
        for k in k_l:
            clust_pipe = make_pipeline(StandardScaler(),  KMeans(n_clusters=k, random_state=conf['seed']))
            clust_pipe.fit(feat_data.loc[tr_idx])
            feat_clust_add[f'clust_{k}'] = clust_pipe.predict(feat_data)
        feat_data_new = feat_data_new.join(feat_clust_add)

    if conf['feat_eng']['add_topic_feat']:
        if conf['feat_eng']['topic_algo'] == 'lsa':
            topic_algo = TruncatedSVD(algorithm = 'randomized', n_components=conf['feat_eng']['add_topic_feat'], random_state=conf['seed'])
        elif conf['feat_eng']['topic_algo'] == 'lda':
            topic_algo = LatentDirichletAllocation(n_components=conf['feat_eng']['add_topic_feat'], random_state=conf['seed'])
        topic_algo.fit(feat_data.loc[tr_idx])
        feat_topic_add = pd.DataFrame(topic_algo.transform(feat_data), index=feat_data.index, columns=topic_algo.get_feature_names_out())   
        feat_data_new = feat_data_new.join(feat_topic_add)
    
    if conf['feat_eng']['add_ind_cols']:
        data['target'] = data['target'].map(lambda x: eval(x))
        data['threat_words'] = data['threat_words'].map(lambda x: eval(x))
        data['threat_words'] = data['threat_words'].apply(lambda x: ' '.join(x))
        
        vec = CountVectorizer(tokenizer=custom_tok, binary=True, min_df=1)
        vec.fit(data.loc[data['train']==1, 'threat_words'])

        feat_ind = pd.DataFrame(vec.transform(data['threat_words']).toarray(), columns=vec.get_feature_names_out())
        feat_data_new = feat_data_new.join(feat_ind)
    
    joblib.dump(feat_data_new, conf['feat_eng']['feat_final_fn'])

    
    
if __name__=='__main__':

    main()