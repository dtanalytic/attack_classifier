import pandas as pd
import numpy as np
import joblib
from nltk.corpus import stopwords
import click
from sklearn.preprocessing import MultiLabelBinarizer
from pandarallel import pandarallel
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


from ruamel.yaml import YAML

import sys
sys.path.append('.')
from src.funcs import preprocess_text
from src.constants import (regexp_email, regexp_cve, regexp_url, regexp_domain, regexp_registry,  regexp_fpath, 
                             regexp_fname, regexp_ipv4, regexp_ipv6,  regex_domain_zone,
                            regexp_hash_md5, regexp_hash_sha1, regexp_hash_sha256, regexp_hash_sha512, regexp_hash_ssdeep, 
                             regexp_coins_eth, regexp_coins_btc, regexp_coins_bch, regexp_coins_ltc,
                            regexp_coins_doge, regexp_coins_dash, regexp_coins_xmr, regexp_coins_neo, regexp_coins_xrp)
from src.spec_funcs import replace_entities

pandarallel.initialize(progress_bar=True, nb_workers=0)

@click.command()
@click.argument('config_path')
def main(config_path):
    
    conf = YAML().load(open(config_path))
    SEED = conf['seed']
    if conf['prep_text']['stop_words']:
        stop_words_l = set(stopwords.words('english'))
    else:
        stop_words_l = []
    
    # data = joblib.load(conf['get_data']['data_fn'])
    data = pd.read_csv(conf['get_data']['data_filt_fn'])
    data['labels'] = data['labels'].map(lambda x: eval(x))
    data['origin_labels'] = data['origin_labels'].map(lambda x: eval(x))

    if conf['prep_text']['replace_entities']:
        # начиная с python 3.7 порядок ключей сохраняется, поэтому можно не упорядочивать
        pat_d = {it:globals()[it] for it in globals() if 'regexp_' in it}
        data['sentence'] = data['sentence'].map(lambda x: replace_entities(x, pat_d))
        
    if conf['prep_text']['labelled_text_only']:
        data = data[data['labels'].str.len()>1]

    # другие названия в квадратных скобках убираем
    data['sentence'] = data['sentence'].str.replace(r'\[(\w+)\]', r'\1', regex=True)

    # после того, как убрал названия в круглых скобках вылезли дубли, например, в отчете та же формулировка, как
    # и в первоисточнике на митр только уже без круглых скобок
    data = data.drop_duplicates(subset=['sentence']) 
    
    # юникод, накоторых токенизатор fastai жаловался
    symb_l = ['\xe4', '\u202f', '\u2192']
    for symb in symb_l:
        data['sentence'] = data['sentence'].str.replace(symb, '')
        
    data['prep_text'] = data['sentence'].apply(lambda x: preprocess_text(x, to_lower=conf['prep_text']['lower'], 
     min_len_token=conf['prep_text']['min_len_token'], max_len_token=conf['prep_text']['max_len_token'], stop_words_l=stop_words_l, 
     stem_lemm_flag=conf['prep_text']['stem_lemm'], save_text=True, save_only_words= conf['prep_text']['save_only_words'], 
        lang='english'))


    ttp_counts_thresh = conf['prep_text']['ttp_counts_thresh']
    ttp_l = data['origin_labels'].explode('origin_labels').value_counts().loc[lambda x: x>ttp_counts_thresh].index.tolist()

    mlb_ttp = MultiLabelBinarizer()
    mlb_ttp.fit([[c] for c in ttp_l+['rare']])
    # mlb_ttp.transform(df.loc[df['origin_labels'].str.len()>1, 'origin_labels'].iloc[:2]).sum()
    joblib.dump(mlb_ttp, conf['prep_text']['ttp_mlb_fn'])


    CLASSES = data.explode('labels')['labels'].dropna().unique()

    mlb = MultiLabelBinarizer(classes=CLASSES)
    mlb.fit([[c] for c in CLASSES])
    joblib.dump(mlb, conf['prep_text']['mlb_fn'])

    data['ttp'] = data['origin_labels'].map(lambda x: [it if it in ttp_l else 'rare' for it in x] )


    data['target_ttp'] = mlb_ttp.transform(data['ttp']).tolist()
    
    val_ts_size = conf['val_ts_size']
    
    mskf = MultilabelStratifiedKFold(n_splits=int(1/(2*val_ts_size)), shuffle=True, random_state=SEED)
    # позиции от 0 до n
    for tr_idx, val_ts_idx in mskf.split(data.values, np.array(data['target_ttp'].tolist())):
        break
    
    mskf = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    
    # позиции от 0 до m
    for val_idx, ts_idx in mskf.split(data.iloc[val_ts_idx].values, np.array(data['target_ttp'].iloc[val_ts_idx].tolist())):
        break
    
    val_idx = val_ts_idx[val_idx]
    ts_idx = val_ts_idx[ts_idx]

    data['split'] = 'tr'
    if conf['val_only_proc']:
        data.loc[(data.index.isin(val_idx)) & ((data['is_proc']==True)|(data['is_proc'].isna())), 'split'] = 'val'
        data.loc[(data.index.isin(ts_idx)) & ((data['is_proc']==True)|(data['is_proc'].isna())), 'split'] = 'ts'
    else:
        data.loc[data.index[val_idx], 'split'] = 'val'
        data.loc[data.index[ts_idx], 'split'] = 'ts'
    
    data.to_csv(conf['prep_text']['prep_fn'], index=False)


if __name__=='__main__':

    main()