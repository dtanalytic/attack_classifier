import pandas as pd
import numpy as np
import joblib
import click

from ruamel.yaml import YAML


import sys
sys.path.append('.')

from src.spec_funcs import calc_select_feat_matr


@click.command()
@click.argument('config_path')
def main(config_path):
    
    conf_ttp = YAML().load(open(config_path))
    conf = YAML().load(open('params.yaml'))

    # чтобы новый конф работал вместо старого в функции
    conf_ttp['feat_gen'] = conf_ttp['feat_gen_ttp'] 
    conf_ttp['seed'] = conf['seed']
    conf_ttp['use_only_proc'] = conf['use_only_proc']
    
    mlb = joblib.load(conf['prep_text']['ttp_mlb_fn'])
    data = pd.read_csv(conf['prep_text']['prep_fn'])

    # тут feat_gen_ttp ключ надо заменить на feat_gen и добавить в  with seed, use_only_proc
    data, feat_data, vec = calc_select_feat_matr(data=data, target_col='ttp', mlb=mlb, conf=conf_ttp)
    

    # joblib.dump(feat_data, conf['feat_gen']['feat_fn'])
    feat_data.to_csv(conf_ttp['feat_gen_ttp']['feat_fn'], index=False)
    data.to_csv(conf_ttp['feat_gen_ttp']['data_fn'], index=False)
    joblib.dump(vec, conf_ttp['feat_gen_ttp']['vec_fn'])
    
    
if __name__=='__main__':

    main()