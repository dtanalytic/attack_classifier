import pandas as pd
import numpy as np
import joblib
import click

from ruamel.yaml import YAML


import sys
sys.path.append('.')
from src.funcs import set_seed

from src.spec_funcs import calc_select_feat_matr
from src.aug_sent import add_aug_sents



@click.command()
@click.argument('config_path')
def main(config_path):
    
    conf_ttp = YAML().load(open(config_path))
    conf = YAML().load(open('params.yaml'))
    
    set_seed(conf['seed'])

    conf_dop = YAML().load(open('dvc_pipes/bert_ttp/params_bert_ttp.yaml'))

    # чтобы новый конф работал вместо старого в функции
    conf_ttp['feat_gen'] = conf_ttp['feat_gen_ttp'] 
    conf_ttp['seed'] = conf['seed']
    conf_ttp['use_only_proc'] = conf['use_only_proc']
    
    mlb = joblib.load(conf['prep_text']['ttp_mlb_fn'])
    
    data = pd.read_csv(conf['prep_text']['prep_fn'])
    

    data['ttp'] = data['ttp'].map(lambda x: eval(x))

    if conf_ttp['feat_gen']['add_aug_sents']:

        data = add_aug_sents(data, conf_ttp, conf_dop['nn_ttp']['maxlen'])
        
    # тут feat_gen_ttp ключ надо заменить на feat_gen и добавить в  with seed, use_only_proc
    data, feat_data, vec = calc_select_feat_matr(data=data, target_col='ttp', mlb=mlb, conf=conf_ttp)
    

    # joblib.dump(feat_data, conf['feat_gen']['feat_fn'])
    feat_data.to_csv(conf_ttp['feat_gen_ttp']['feat_fn'], index=False)
    data.to_csv(conf_ttp['feat_gen_ttp']['data_fn'], index=False)
    joblib.dump(vec, conf_ttp['feat_gen_ttp']['vec_fn'])
    
    
if __name__=='__main__':

    main()