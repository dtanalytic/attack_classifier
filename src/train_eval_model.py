import pandas as pd
import numpy as np
import joblib
import json
import click
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import ClassifierChain
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


from ruamel.yaml import YAML

import sys
sys.path.append('.')

def metric_multi(y, y_pred, metric_fn, **kwargs):
  metric_l = []
  for i in range(y.shape[1]):
    if len(set(y[:, i]))!=1:
        metric = metric_fn(y[:, i], y_pred[:,i], **kwargs)
        metric_l.append(metric)

  return np.mean(metric_l), metric_l



@click.command()
@click.argument('config_path')
def main(config_path):
    
    conf = YAML().load(open(config_path))
    np.random.seed(conf['seed'])
    
    feat_data = joblib.load(conf['feat_eng']['feat_final_fn'])
    mlb = joblib.load(conf['prep_text']['mlb_fn'])
    data = pd.read_csv(conf['feat_gen']['data_fn'])
    
    data['labels'] = data['labels'].map(lambda x: eval(x))
    
    tr_idx = data.query('train==1').index
    val_idx = data.query('valid==1').index
    ts_idx = data.query('test==1').index
    
    
    if conf['train_eval_model']['chain']:
        wrap_class = ClassifierChain
    else:
        wrap_class = OneVsRestClassifier

        
    if conf['train_eval_model']['model']=='logreg':
        model = LogisticRegression(random_state=conf['seed'], class_weight=conf['train_eval_model']['balanced'], max_iter=1000)
    elif conf['train_eval_model']['model']=='tree':
        model = DecisionTreeClassifier(random_state=conf['seed'], class_weight=conf['train_eval_model']['balanced'])
    elif conf['train_eval_model']['model']=='boost':
        model = HistGradientBoostingClassifier(random_state=conf['seed'], class_weight=conf['train_eval_model']['balanced'])
    elif conf['train_eval_model']['model']=='knn':
        model = KNeighborsClassifier(class_weight=conf['train_eval_model']['balanced'])    
    elif conf['train_eval_model']['model']=='forest':
        model = ExtraTreesClassifier(random_state=conf['seed'], class_weight=conf['train_eval_model']['balanced'])

        
    clf_pipe = make_pipeline(RobustScaler(), wrap_class(model, random_state=conf['seed'])) if conf['train_eval_model']['chain'] else make_pipeline(RobustScaler(), wrap_class(model))
    Y_train = mlb.transform(data.loc[tr_idx, 'labels'])
    clf_pipe.fit(feat_data.loc[tr_idx], Y_train)
    
    Y_tr_proba = clf_pipe.predict_proba(feat_data.loc[tr_idx])
    
    tr_roc_auc, _ = metric_multi(Y_train, Y_tr_proba, roc_auc_score)
    tr_logloss, _ = metric_multi(Y_train, Y_tr_proba, log_loss, labels=[0,1])
    tr_pr_auc, _ = metric_multi(Y_train, Y_tr_proba, average_precision_score)
    
    Y_proba = clf_pipe.predict_proba(feat_data.loc[val_idx])
    Y_true = mlb.transform(data.loc[val_idx, 'labels'])
    
    roc_auc, _ = metric_multi(Y_true, Y_proba, roc_auc_score)
    logloss, _ = metric_multi(Y_true, Y_proba, log_loss, labels=[0,1])
    pr_auc, _ = metric_multi(Y_true, Y_proba, average_precision_score)

    with open(conf['train_eval_model']['add_metrics_fn'], 'wt') as f_wr:
        json.dump({'val_roc_auc':roc_auc, 'val_logloss':logloss, 'tr_roc_auc':tr_roc_auc, 'tr_logloss':tr_logloss}, f_wr)
        
    with open(conf['train_eval_model']['metrics_fn'], 'wt') as f_wr:
        json.dump({'val_pr_auc':pr_auc, 'val_logloss':logloss, 'tr_pr_auc':tr_pr_auc, 'tr_logloss':tr_logloss}, f_wr)

    joblib.dump(clf_pipe, conf['train_eval_model']['model_fn'])
    
if __name__=='__main__':

    main()