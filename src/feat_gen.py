import pandas as pd
import numpy as np
import joblib
import click
from nltk import word_tokenize
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from ruamel.yaml import YAML

import sys
sys.path.append('.')

# если лямбду сделать, то дамп не получится сделать
def custom_tok(x):
    return word_tokenize(x)
    
@click.command()
@click.argument('config_path')
def main(config_path):
    
    conf = YAML().load(open(config_path))
    np.random.seed(conf['seed'])

    SEED = conf['seed']
    
    mlb = joblib.load(conf['prep_text']['mlb_fn'])
    data = pd.read_csv(conf['prep_text']['prep_fn'])
    
    data['labels'] = data['labels'].map(lambda x: eval(x))
    data['target'] = mlb.transform(data['labels']).tolist()


    val_ts_size = conf['val_ts_size']
    
    mskf = MultilabelStratifiedKFold(n_splits=int(2/val_ts_size), shuffle=True, random_state=SEED)
    
    # позиции от 0 до n
    for tr_idx, val_ts_idx in mskf.split(data.values, np.array(data['target'].tolist())):
        break
    
    mskf = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    
    # позиции от 0 до m
    for val_idx, ts_idx in mskf.split(data.iloc[val_ts_idx].values, np.array(data['target'].iloc[val_ts_idx].tolist())):
        break
    
    val_idx = val_ts_idx[val_idx]
    ts_idx = val_ts_idx[ts_idx]
    
    data['split'] = 'tr'
    data.loc[data.index[val_idx], 'split'] = 'val'
    data.loc[data.index[ts_idx], 'split'] = 'ts'

    # добавление новых признаков
    # здесь же можно подумать о чистке    
    data['threat_words'] = data['sentence'].str.findall('\((T[\d\.]+)\)')
    data['prep_text'] = data[['prep_text', 'threat_words']].apply(lambda x: ' '.join(x['threat_words']) + f' {x["prep_text"]}' , axis=1)
    
    
    if conf['feat_gen']['feat_strategy'] == 'tfidf':
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(tokenizer=custom_tok, binary=conf['feat_gen']['binary'], max_features=conf['feat_gen']['max_features'])
    elif conf['feat_gen']['feat_strategy'] == 'count':
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer(tokenizer=custom_tok , binary=conf['feat_gen']['binary'], max_features=conf['feat_gen']['max_features'])
    else:
        print(f"mistake in {conf['feat_gen']['feat_strategy']}, must be in (count, tfidf)")
    
    vec.fit(data.query('split=="tr"')['prep_text'])

    feat_data = pd.DataFrame(vec.transform(data['prep_text']).toarray(), columns=vec.get_feature_names_out(), index=data.index)
    
    # data['features'] = feat_data.to_numpy().tolist()
    
    joblib.dump(feat_data, conf['feat_gen']['feat_fn'])
    data.to_csv(conf['feat_gen']['data_fn'], index=False)
    joblib.dump(vec, conf['feat_gen']['vec_fn'])
    
    
if __name__=='__main__':

    main()