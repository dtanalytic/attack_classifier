import pandas as pd
import numpy as np
import joblib
import click
from nltk import word_tokenize
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from itertools import chain

from ruamel.yaml import YAML

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import RobustScaler

import sys
sys.path.append('.')

# если лямбду сделать, то дамп не получится сделать
def custom_tok(x):
    return word_tokenize(x)

def add_new_feat(data, feat_full_data, cls, seed, tr_idx, col_l, feat_thresh = 0.8):    
    
    cls_filter = (data['labels'].map(lambda x: cls in x))
    
    feat_full_data[ 'y'] = 0
    feat_full_data.loc[cls_filter, 'y'] = 1
    
    feat_full_data['y'] = feat_full_data['y'].astype(int)
    
    cls = DecisionTreeClassifier(max_depth=10, random_state=seed)
    
    cls.fit(feat_full_data.drop(columns='y').loc[tr_idx], feat_full_data['y'].loc[tr_idx])
    
    imps_df = pd.DataFrame({'feat':feat_full_data.drop(columns='y').columns, 'imp':cls.feature_importances_})\
        .sort_values(by='imp', ascending=False)
    
    idx = np.where(imps_df['imp'].cumsum()/imps_df['imp'].sum()>feat_thresh)[0].min()
    
    return [it for it in imps_df.loc[imps_df.index[:idx], 'feat'].tolist() if not it in col_l]



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
    
    mskf = MultilabelStratifiedKFold(n_splits=int(1/(2*val_ts_size)), shuffle=True, random_state=SEED)
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
        
        vec = TfidfVectorizer(tokenizer=custom_tok, binary=conf['feat_gen']['binary'], 
                      max_features=conf['feat_gen']['max_features'], min_df=conf['feat_gen']['min_df'], 
                      max_df=conf['feat_gen']['max_df'], ngram_range=(1,3))
        
    elif conf['feat_gen']['feat_strategy'] == 'count':
        from sklearn.feature_extraction.text import CountVectorizer
        vec = CountVectorizer(tokenizer=custom_tok , binary=conf['feat_gen']['binary'], max_features=conf['feat_gen']['max_features'], min_df=conf['feat_gen']['min_df'], max_df=conf['feat_gen']['max_df'], ngram_range=(1,3))
    else:
        print(f"mistake in {conf['feat_gen']['feat_strategy']}, must be in (count, tfidf)")
    
    vec.fit(data.query('split=="tr"')['prep_text'])

    feat_data = pd.DataFrame(vec.transform(data['prep_text']).toarray(), columns=vec.get_feature_names_out(), index=data.index)

    
    # выбор признаков путем отбора двумя алгоритмами
    col_l = feat_data.columns
    
    feat_data = feat_data.assign(y = data['target'])
    
    Y_train = np.array(feat_data.loc[tr_idx, 'y'].values.tolist())
    X_train = feat_data.loc[tr_idx].drop(columns='y').astype(np.float16).values
    
    # выбираем логрегом    
    model = LogisticRegression(penalty='l1', solver='liblinear', C = conf['feat_gen']['logreg_c'],
                              random_state=conf['seed'], max_iter=10000)
    
    feat_sel_model = OneVsRestClassifier(model)
    
    feat_sel_model.fit(X_train, Y_train)

    coef_ar = np.vstack([estimator.coef_ for estimator in feat_sel_model.estimators_])
    feat_coef_ar = coef_ar.mean(axis=0)
    feat_idx = np.where((feat_coef_ar>feat_coef_ar.mean()))[0].tolist()
    
    imp_logreg_df = pd.DataFrame({'features':np.array(col_l)[feat_idx], 'imp':feat_coef_ar[feat_idx]}).sort_values(by='imp', ascending=False)

    # выбираем деревом
    model = DecisionTreeClassifier(random_state=conf['seed'], max_depth=conf['feat_gen']['tree_maxdepth'])
    feat_sel_model = OneVsRestClassifier(model)
    
    feat_sel_model.fit(X_train, Y_train)
    
    coef_ar = np.vstack([estimator.feature_importances_ for estimator in feat_sel_model.estimators_])
    
    feat_coef_ar = coef_ar.mean(axis=0)
    feat_idx = np.where((feat_coef_ar>feat_coef_ar.mean()))[0].tolist()
    
    imp_tree_df = pd.DataFrame({'features':np.array(col_l)[feat_idx], 'imp':feat_coef_ar[feat_idx]}).sort_values(by='imp', ascending=False)
    
    
    # объединяем
    imp_df = imp_tree_df.merge(imp_logreg_df, how='outer', on='features', suffixes=('_tree', '_logreg')).dropna()
    sc = RobustScaler()    
    imp_df[['imp_tree', 'imp_logreg']] = sc.fit_transform(imp_df[['imp_tree', 'imp_logreg']])
    imp_df = imp_df.assign(imp = imp_df[['imp_tree', 'imp_logreg']].mean(axis=1)).sort_values(by='imp', ascending=False)
    feat_cols = imp_df['features'].values
    final_cols = feat_cols.tolist()

    if conf['feat_gen']['add_individual_thresh']:        
        feat_d = {}
        for cls in mlb.classes_:
            feat_d[cls] = add_new_feat(data, feat_data, cls, conf['seed'], tr_idx, final_cols, feat_thresh=conf['feat_gen']['add_individual_thresh'])

        col_add_l = list(chain(*feat_d.values()))

        final_cols = final_cols +  col_add_l
        
    feat_data = feat_data[final_cols]

    
    # joblib.dump(feat_data, conf['feat_gen']['feat_fn'])
    feat_data.to_csv(conf['feat_gen']['feat_fn'], index=False)
    data.to_csv(conf['feat_gen']['data_fn'], index=False)
    joblib.dump(vec, conf['feat_gen']['vec_fn'])
    
    
if __name__=='__main__':

    main()