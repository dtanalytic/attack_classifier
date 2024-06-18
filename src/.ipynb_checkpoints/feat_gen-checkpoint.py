import pandas as pd
import numpy as np
import joblib
import click
from nltk import word_tokenize
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
    mlb = joblib.load(conf['prep_text']['mlb_fn'])
    data = pd.read_csv(conf['prep_text']['prep_fn'])
    
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    tr_idx, val_idx, ts_idx = np.split(idx, [round(0.8*len(idx)), round(0.9*len(idx))])

    data.loc[data.index[tr_idx], 'train'] = 1
    data.loc[data.index[val_idx], 'valid'] = 1
    data.loc[data.index[ts_idx], 'test'] = 1
    data[['train', 'valid', 'test']] = data[['train', 'valid', 'test']].fillna(0)
    
    # добавление новых признаков
    # здесь же можно подумать о чистке    
    data['threat_words'] = data['sentence'].str.findall('\((T[\d\.]+)\)')
    # data['prep_text'] = data[['prep_text', 'threat_words']].apply(lambda x: ' '.join(x['threat_words']) + f' {x["prep_text"]}' , axis=1)

    
    if conf['feat_gen']['feat_strategy'] == 'tfidf':
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(tokenizer=custom_tok, binary=conf['feat_gen']['binary'], max_features=conf['feat_gen']['max_features'])
    elif conf['feat_gen']['feat_strategy'] == 'count':
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer(tokenizer=custom_tok , binary=conf['feat_gen']['binary'], max_features=conf['feat_gen']['max_features'])
    else:
        print(f"mistake in {conf['feat_gen']['feat_strategy']}, must be in (count, tfidf)")
    
    vec.fit(data.iloc[tr_idx]['prep_text'])

    feat_data = pd.DataFrame(vec.transform(data['prep_text']).toarray(), columns=vec.get_feature_names_out(), index=data.index)
    
    # data['features'] = feat_data.to_numpy().tolist()
    
    joblib.dump(feat_data, conf['feat_gen']['feat_fn'])
    data.to_csv(conf['feat_gen']['data_fn'], index=False)
    joblib.dump(vec, conf['feat_gen']['vec_fn'])
    
    
if __name__=='__main__':

    main()