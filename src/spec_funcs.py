import numpy as np
import pandas as pd
import joblib
import json

from nltk import word_tokenize
from itertools import chain
import re 

from sklearn.cluster import KMeans


from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD, NMF
from sklearn.feature_extraction.text import CountVectorizer


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.pipeline import make_pipeline
from sklearn.multioutput import ClassifierChain
from sklearn.metrics import (log_loss, roc_auc_score, average_precision_score, f1_score, 
                            precision_recall_fscore_support, confusion_matrix)
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from src.funcs import metric_multi
from src.funcs import get_conf_df, get_pred_thresh, set_seed, get_opt_thresh


# если лямбду сделать, то дамп не получится сделать
def custom_tok(x):
    return word_tokenize(x)

def replace_entities(s, pat_d):
    
    w_l = s.split()
    for key, val in pat_d.items():    
        w_l = [w if not re.search(val, w.lower()) else re.sub(val, key, w.lower()) for w in w_l]
    
    return ' '.join(w_l)


def train_eval_classic(conf, target_col, fig_size1, fig_size2, thresh_space_l=[]):

    if target_col=='ttp':
        mlb = joblib.load(conf['prep_text']['ttp_mlb_fn'])
    else:
        mlb = joblib.load(conf['prep_text']['mlb_fn'])
    data = pd.read_csv(conf['feat_gen']['data_fn'])
    # feat_data = pd.read_csv(conf['feat_gen']['feat_fn'])
    feat_data = pd.read_csv(conf['feat_eng']['feat_final_fn'])
    
    data['target'] = data['target'].map(lambda x: eval(x))
    data[target_col] = data[target_col].map(lambda x: eval(x))
    
    tr_idx = data.query('split=="tr"').index
    val_idx = data.query('split=="val"').index
    ts_idx = data.query('split=="ts"').index
    
    # определяемся с классификатором атак
    if conf['train_eval_model']['attack_clf']:
        
        data['is_attack'] = (data[target_col].str.len()>0).astype(np.int8)
        attack_clf = make_pipeline(RobustScaler(), 
                                 LogisticRegression(random_state=conf['seed'], class_weight=conf['train_eval_model']['balanced'], max_iter=10000))
        
        Y_at_train = np.array(data.loc[tr_idx, 'is_attack'].values.tolist())
        
        attack_clf.fit(feat_data.loc[tr_idx].values, Y_at_train)
        
        Y_at_tr_proba = attack_clf.predict_proba(feat_data.loc[tr_idx])[:,1]
        
        Y_at_val_proba = attack_clf.predict_proba(feat_data.loc[val_idx])[:, 1]
        Y_at_val = np.array(data.loc[val_idx, 'is_attack'].values.tolist())
    

        metric = conf['train_eval_model']['opt_metric']
        res_df, thresh = get_pred_thresh(Y_at_val, Y_at_val_proba, opt_metric=metric)
        feat_data['attack_pred'] = (attack_clf.predict_proba(feat_data)[:,1]>thresh).astype(int)
        feat_data.to_csv(conf['feat_eng']['feat_final_fn'], index=False)

    if conf['train_eval_model']['chain']:
        wrap_class = ClassifierChain
    else:
        wrap_class = OneVsRestClassifier 
        
    if conf['train_eval_model']['model']=='logreg':
        model = LogisticRegression(random_state=conf['seed'], class_weight=conf['train_eval_model']['balanced'], max_iter=10000)
    elif conf['train_eval_model']['model']=='tree':
        model = DecisionTreeClassifier(random_state=conf['seed'], class_weight=conf['train_eval_model']['balanced'])
    elif conf['train_eval_model']['model']=='boost':
        model = HistGradientBoostingClassifier(random_state=conf['seed'], class_weight=conf['train_eval_model']['balanced'])
    elif conf['train_eval_model']['model']=='knn':
        model = KNeighborsClassifier(class_weight=conf['train_eval_model']['balanced'])    
    elif conf['train_eval_model']['model']=='forest':
        model = ExtraTreesClassifier(random_state=conf['seed'], class_weight=conf['train_eval_model']['balanced'])
    
        
    clf_pipe = make_pipeline(RobustScaler(), wrap_class(model, random_state=conf['seed'])) if conf['train_eval_model']['chain'] else make_pipeline(RobustScaler(), wrap_class(model))
    
    Y_train = np.array(data.loc[tr_idx, 'target'].values.tolist())
    
    clf_pipe.fit(feat_data.loc[tr_idx].values, Y_train)

    Y_tr_proba = clf_pipe.predict_proba(feat_data.loc[tr_idx])    

    tr_roc_auc, _ = metric_multi(Y_train, Y_tr_proba, roc_auc_score)
    tr_logloss, _ = metric_multi(Y_train, Y_tr_proba, log_loss, labels=[0,1])
    tr_pr_auc, _ = metric_multi(Y_train, Y_tr_proba, average_precision_score)
    
    Y_val_proba = clf_pipe.predict_proba(feat_data.loc[val_idx])
    Y_val = np.array(data.loc[val_idx, 'target'].values.tolist())
    
    
    roc_auc, _ = metric_multi(Y_val, Y_val_proba, roc_auc_score)
    logloss, _ = metric_multi(Y_val, Y_val_proba, log_loss, labels=[0,1])
    pr_auc, _ = metric_multi(Y_val, Y_val_proba, average_precision_score)
    
    
    with open(conf['train_eval_model']['metrics_fn'], 'wt') as f_wr:
        json.dump({'val_pr_auc':pr_auc, 'val_logloss':logloss, 'tr_pr_auc':tr_pr_auc, 'tr_logloss':tr_logloss}, f_wr)

    joblib.dump(clf_pipe, conf['train_eval_model']['model_fn'])

    # import pdb;pdb.set_trace()
    thresh_l = get_opt_thresh(y_true = Y_val, probas = Y_val_proba, mlb = mlb, opt_metric=conf['train_eval_model']['opt_metric'], thresh_space_l=thresh_space_l,
                          dump_fn = conf['train_eval_model']['opt_metric_fn'])

    preds = np.apply_along_axis(lambda x: x>=x[-1], 0, np.vstack([Y_tr_proba, thresh_l]))[:-1].astype(int)

    if conf['train_eval_model']['attack_clf']=='alone':
        attack_idx = np.where((feat_data.loc[data.index[tr_idx], 'attack_pred']).values)[0]
        preds[~attack_idx, :] = 0
    
    
    p_tr_micro, r_tr_micro, f1_tr_micro, _ = precision_recall_fscore_support(Y_train, (preds).astype(int), average='micro')
    p_tr_macro, r_tr_macro, f1_tr_macro, _ = precision_recall_fscore_support(Y_train, (preds).astype(int), average='macro')

    res_df = pd.DataFrame()
    res_df['y'] = Y_val.tolist()
    res_df['y_proba'] = Y_val_proba.tolist()
    
    thresh_col = 'y_p'
    res_df[thresh_col] = res_df['y_proba'].map(lambda x: [int(val>=thresh) for val, thresh in zip(x, thresh_l)])
    
    if conf['train_eval_model']['attack_clf']=='alone':
        attack_idx = np.where((feat_data.loc[data.index[val_idx], 'attack_pred']).values)[0]
        thresh_ar = np.array(res_df[thresh_col].values.tolist())
        thresh_ar[~attack_idx, :] = 0
        res_df[thresh_col] = thresh_ar.tolist()
    
    error_df = data.loc[val_idx].reset_index(names='val_idx').join(res_df[['y', 'y_proba', thresh_col]])
    error_df['prob_label'] = mlb.inverse_transform(np.array(error_df[thresh_col].tolist()))
    
    p_val_micro, r_val_micro, f1_val_micro, sup = precision_recall_fscore_support(np.array(res_df['y'].values.tolist()), 
                                                        np.array(res_df[thresh_col].values.tolist()), average='micro')
    p_val_macro, r_val_macro, f1_val_macro, sup = precision_recall_fscore_support(np.array(res_df['y'].values.tolist()), 
                                                        np.array(res_df[thresh_col].values.tolist()), average='macro')

    _, res_l = metric_multi(np.array(error_df['y'].tolist()), np.array(error_df[thresh_col].tolist()), f1_score)

    pd.DataFrame({'qual':res_l, 'class':mlb.classes_}).sort_values(by='qual').to_csv(conf['train_eval_model']['by_class_metric_fn'], index=False)

    conf_df = get_conf_df(error_df, target_col=target_col)

    conf_df.to_csv(conf['train_eval_model']['conf_df_fn'], index=False)


    plt.figure(figsize=fig_size1)

    labels = np.sort(conf_df[target_col].unique())
    cm = confusion_matrix(y_true = conf_df[target_col], y_pred = conf_df['prob_label'], 
                          labels=labels, normalize=None)
    
    draw_df = pd.concat([pd.DataFrame(cm, index=labels, columns=labels),
                        pd.DataFrame([cm.sum(axis=0)], columns=labels, index=['pred_sum'])])\
                .assign(true_sum=lambda x: x.sum(axis=1))
    
    sns.heatmap(draw_df, annot=True, fmt='d', cmap='viridis', cbar=False)
    
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.savefig(conf['train_eval_model']['conf_matrix_fn'])

    er_df = conf_df.query(f'{target_col}!=prob_label and prob_label!="empty" and {target_col}!="empty"')
    if target_col == 'ttp':
        er_df = er_df.query(f'{target_col}!= "rare" and prob_label!="rare"')
    
    heat_df = er_df.assign(val=1).groupby([target_col, 'prob_label']).sum().loc[lambda x: x['val']>1]['val'].unstack().fillna(0)

    plt.figure(figsize=fig_size2)
    
    sns.heatmap(heat_df, annot=True, cbar=False)
    plt.savefig(conf['train_eval_model']['conf_matrix_main_er_fn'])

    with open(conf['train_eval_model']['add_metrics_fn'], 'wt') as f_wr:
        json.dump({'val_roc_auc':roc_auc, 'val_logloss':logloss, 'tr_roc_auc':tr_roc_auc, 'tr_logloss':tr_logloss,
                  'p_tr_micro':p_tr_micro, 'r_tr_micro':r_tr_micro, 'f1_tr_micro':f1_tr_micro,
                   'p_tr_macro':p_tr_macro, 'r_tr_macro':r_tr_macro, 'f1_tr_macro':f1_tr_macro,
                   'p_val_micro':p_val_micro, 'r_val_micro':r_val_micro, 'f1_val_micro':f1_val_micro,
                   'p_val_macro':p_val_macro, 'r_val_macro':r_val_macro, 'f1_val_macro':f1_val_macro
                  }, f_wr)

def calc_feat_eng_matr(conf, target_col):
    
    if target_col=='ttp':
        mlb = joblib.load(conf['prep_text']['ttp_mlb_fn'])
    else:
        mlb = joblib.load(conf['prep_text']['mlb_fn'])
    
    data = pd.read_csv(conf['feat_gen']['data_fn'])
    # feat_data = joblib.load(conf['feat_gen']['feat_fn'])
    feat_data = pd.read_csv(conf['feat_gen']['feat_fn'])
    
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
        # add word feat for adversaries 
        feat_data_new = feat_data
        selection_adversary = data['sentence'].str.contains('versar')
        feat_data_new.loc[selection_adversary, 'adversary'] = 1
        feat_data_new['adversary'] = feat_data_new['adversary'].fillna(0)
        
        # data['target'] = data['target'].map(lambda x: eval(x))
        # data['threat_words'] = data['threat_words'].map(lambda x: eval(x))
        # data['threat_words'] = data['threat_words'].apply(lambda x: ' '.join(x))
        
        # vec = CountVectorizer(tokenizer=custom_tok, binary=True, min_df=1)
        # vec.fit(data.loc[data['split']=='tr', 'threat_words'])
    
        # feat_ind = pd.DataFrame(vec.transform(data['threat_words']).toarray(), columns=vec.get_feature_names_out())
        # feat_data_new = feat_data_new.join(feat_ind)
    
    # joblib.dump(feat_data_new, conf['feat_eng']['feat_final_fn'])
    feat_data_new.to_csv(conf['feat_eng']['feat_final_fn'], index=False)


def calc_select_feat_matr(data, target_col, mlb, conf):
    
    np.random.seed(conf['seed'])
    SEED = conf['seed']
    
    
    data['target'] = mlb.transform(data[target_col]).tolist()
    
    
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
            feat_d[cls] = add_new_feat(data, feat_data, cls, conf['seed'], tr_idx, final_cols, target_col, sel_estim_num=conf['feat_gen']['ind_sel_estim_num'], sel_tree_max_depth=conf['feat_gen']['ind_sel_tree_max_depth'],feat_thresh=conf['feat_gen']['add_individual_thresh'])

        col_add_l = list(chain(*feat_d.values()))

        final_cols = final_cols +  col_add_l
        
    final_cols = pd.Series(final_cols).drop_duplicates().tolist()
    feat_data = feat_data[final_cols]
    
    return data, feat_data, vec

    
def add_new_feat(data, feat_full_data, cls, seed, tr_idx, col_l, target_col, sel_estim_num, sel_tree_max_depth, feat_thresh = 0.8):    
    
    cls_filter = (data[target_col].map(lambda x: cls in x))
    
    feat_full_data[ 'y'] = 0
    feat_full_data.loc[cls_filter, 'y'] = 1
    
    feat_full_data['y'] = feat_full_data['y'].astype(int)
    
    # cls = DecisionTreeClassifier(max_depth=10, random_state=seed)
    cls = RandomForestClassifier(n_estimators=sel_estim_num, random_state=seed, max_depth=sel_tree_max_depth)
    
    cls.fit(feat_full_data.drop(columns='y').loc[tr_idx], feat_full_data['y'].loc[tr_idx])
    
    imps_df = pd.DataFrame({'feat':feat_full_data.drop(columns='y').columns, 'imp':cls.feature_importances_})\
        .sort_values(by='imp', ascending=False)
    
    idx = np.where(imps_df['imp'].cumsum()/imps_df['imp'].sum()>feat_thresh)[0].min()
    
    return [it for it in imps_df.loc[imps_df.index[:idx], 'feat'].tolist() if not it in col_l]
