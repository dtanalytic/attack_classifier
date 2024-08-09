import pandas as pd
import numpy as np
import joblib
import json
import click
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (log_loss, roc_auc_score, average_precision_score, f1_score, 
                            precision_recall_fscore_support, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

import seaborn as sns
import matplotlib.pyplot as plt

from ruamel.yaml import YAML

import sys
sys.path.append('.')

def metric_multi(y, y_pred, metric_fn, ignore_const_target=True, **kwargs):
  metric_l = []
  for i in range(y.shape[1]):
    if ignore_const_target:
        # if len(set(y[:, i]))!=1:
        if set(y[:, i])!={0}:
            metric = metric_fn(y[:, i], y_pred[:,i], **kwargs)
            metric_l.append(metric)
        elif (y_pred[:, i]>0.5).sum()>0:
            metric_l.append(0)
        else:
            metric_l.append(1)
    else:
        metric = metric_fn(y[:, i], y_pred[:,i], labels=[0,1], **kwargs)
        metric_l.append(metric)


  return np.mean(metric_l), metric_l



@click.command()
@click.argument('config_path')
def main(config_path):
    
    conf = YAML().load(open(config_path))
    np.random.seed(conf['seed'])
    
    # feat_data = joblib.load(conf['feat_eng']['feat_final_fn'])
    feat_data = pd.read_csv(conf['feat_eng']['feat_final_fn'])
    
    mlb = joblib.load(conf['prep_text']['mlb_fn'])
    data = pd.read_csv(conf['feat_gen']['data_fn'])
    
    data['target'] = data['target'].map(lambda x: eval(x))
    data['labels'] = data['labels'].map(lambda x: eval(x))
    tr_idx = data.query('split=="tr"').index
    val_idx = data.query('split=="val"').index
    ts_idx = data.query('split=="ts"').index
    
    
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

    res_df = pd.DataFrame()
    res_df['y'] = Y_val.tolist()
    res_df['y_proba'] = Y_val_proba.tolist()


    

    # подберем границу
    y_true = Y_train
    probas = Y_tr_proba
    thresh_l = []
    res_d = {}

    num_cls = len(mlb.classes_)
    metric = conf['train_eval_model']['opt_metric']
    for i in range(num_cls):
        res_metrics_l = []
        for thresh in np.arange(0.05, 0.95, 0.05):
            
            y_i = y_true[:,i]
            proba_i = (probas[:,i]>thresh).astype(int)
            p, r, f1, sup = [it[1] for it in precision_recall_fscore_support(y_i, proba_i)]
    
            res_metrics_df = pd.DataFrame({'precision':p, 'recall':r, 'f1':f1, 'sup':sup}, index=[thresh])
            res_metrics_l.append(res_metrics_df)
    
    
        res_d[i] = pd.concat([it for it in res_metrics_l], axis=0)
        thresh_l.append(res_d[i].index[res_d[i][metric].argmax()].round(3))
    
    pd.concat([v.assign(class_nm=mlb.classes_[k]) for k,v in res_d.items()], ignore_index=True).to_csv(conf['train_eval_model']['opt_metric_fn'], index=False)
    


    preds = np.apply_along_axis(lambda x: x>=x[-1], 0, np.vstack([probas, thresh_l]))[:-1].astype(int)

    p_tr_micro, r_tr_micro, f1_tr_micro, _ = precision_recall_fscore_support(y_true, (preds).astype(int), average='micro')
    p_tr_macro, r_tr_macro, f1_tr_macro, _ = precision_recall_fscore_support(y_true, (preds).astype(int), average='macro')

    thresh_col = 'y_p'
    res_df[thresh_col] = res_df['y_proba'].map(lambda x: [int(val>=thresh) for val, thresh in zip(x, thresh_l)])


    error_df = data.loc[val_idx].reset_index(names='val_idx').join(res_df[['y', 'y_proba', thresh_col]])
    error_df['prob_label'] = mlb.inverse_transform(np.array(error_df[thresh_col].tolist()))

    p_val_micro, r_val_micro, f1_val_micro, sup = precision_recall_fscore_support(np.array(res_df['y'].values.tolist()), 
                                                        np.array(res_df[thresh_col].values.tolist()), average='micro')
    p_val_macro, r_val_macro, f1_val_macro, sup = precision_recall_fscore_support(np.array(res_df['y'].values.tolist()), 
                                                        np.array(res_df[thresh_col].values.tolist()), average='macro')

    _, res_l = metric_multi(np.array(error_df['y'].tolist()), np.array(error_df[thresh_col].tolist()), f1_score)
    
    pd.DataFrame({'qual':res_l, 'class':mlb.classes_}).sort_values(by='qual').to_csv(conf['train_eval_model']['by_class_metric_fn'], index=False)


    # считаем матрицу расхождений
    df_l = []
    for idx in range(error_df[['labels', 'prob_label']].shape[0]):
        row = error_df[['labels', 'prob_label']].iloc[idx]
        row_l = []
        good_s = set(row['labels']).intersection(row['prob_label'])
        if len(good_s)>0:
            for it in good_s:
                row_l.append(pd.DataFrame({'labels':it, 'prob_label':it}, index=[idx]))
                
        in_labels_s = set(row['labels']).difference(row['prob_label'])
        in_probas_s = set(row['prob_label']).difference(row['labels'])
        l1 = len(in_labels_s)
        l2 = len(in_probas_s)
    
        # if l1>0 or l2>0:
        row_l.append(pd.DataFrame({'labels':[in_labels_s], 'prob_label':[in_probas_s]}, index=[idx]))
    
        df_l.append(pd.concat(row_l))
        
    conf_df = pd.concat(df_l).explode('labels').explode('prob_label').fillna('empty').reset_index()

    conf_df.to_csv(conf['train_eval_model']['conf_df_fn'], index=False)
    
    plt.figure(figsize=(15,8))
    
    labels = np.sort(conf_df['labels'].unique())
    cm = confusion_matrix(y_true = conf_df['labels'], y_pred = conf_df['prob_label'], 
                          labels=labels,
                          normalize=None)
    
    draw_df = pd.concat([pd.DataFrame(cm, index=labels, columns=labels),
                        pd.DataFrame([cm.sum(axis=0)], columns=labels, index=['pred_sum'])])\
                .assign(true_sum=lambda x: x.sum(axis=1))
    
    sns.heatmap(draw_df, annot=True, fmt='d', cmap='viridis', cbar=False)
    
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(conf['train_eval_model']['conf_matrix_fn'])


    with open(conf['train_eval_model']['add_metrics_fn'], 'wt') as f_wr:
        json.dump({'val_roc_auc':roc_auc, 'val_logloss':logloss, 'tr_roc_auc':tr_roc_auc, 'tr_logloss':tr_logloss,
                  'p_tr_micro':p_tr_micro, 'r_tr_micro':r_tr_micro, 'f1_tr_micro':f1_tr_micro,
                   'p_tr_macro':p_tr_macro, 'r_tr_macro':r_tr_macro, 'f1_tr_macro':f1_tr_macro,
                   'p_val_micro':p_val_micro, 'r_val_micro':r_val_micro, 'f1_val_micro':f1_val_micro,
                   'p_val_macro':p_val_macro, 'r_val_macro':r_val_macro, 'f1_val_macro':f1_val_macro
                  }, f_wr)


if __name__=='__main__':

    main()