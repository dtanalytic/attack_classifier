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
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import RobustScaler

import sys
sys.path.append('.')

from src.funcs import add_new_feat
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

    # conf, mlb different, target_column
    if conf['use_only_proc']:
        data = data[(data['is_proc']==True)|(data['is_proc'].isna())].reset_index(drop=True)

    # добавление новых признаков
    # здесь же можно подумать о чистке    
    data['threat_words'] = data['sentence'].str.findall('\((T[\d\.]+)\)')
    data['prep_text'] = data[['prep_text', 'threat_words']].apply(lambda x: ' '.join(x['threat_words']) + f' {x["prep_text"]}' , axis=1)


    tr_idx = data.query('split=="tr"').index
    val_idx = data.query('split=="val"').index
    ts_idx = data.query('split=="ts"').index
    
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
    # model = DecisionTreeClassifier(random_state=conf['seed'], max_depth=conf['feat_gen']['tree_maxdepth'])
    model = RandomForestClassifier(n_estimators=100, random_state=conf['seed'], max_depth=conf['feat_gen']['tree_maxdepth'])
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
            feat_d[cls] = add_new_feat(data, feat_data, cls, conf['seed'], tr_idx, final_cols, sel_tree_max_depth=conf['feat_gen']['ind_sel_tree_max_depth'],feat_thresh=conf['feat_gen']['add_individual_thresh'])

        col_add_l = list(chain(*feat_d.values()))

        final_cols = final_cols +  col_add_l
        
    final_cols = pd.Series(final_cols).drop_duplicates().tolist()
    feat_data = feat_data[final_cols]


    
    # joblib.dump(feat_data, conf['feat_gen']['feat_fn'])
    feat_data.to_csv(conf['feat_gen']['feat_fn'], index=False)
    data.drop(columns=['ttp', 'target_ttp']).to_csv(conf['feat_gen']['data_fn'], index=False)
    joblib.dump(vec, conf['feat_gen']['vec_fn'])
    
    
if __name__=='__main__':

    main()