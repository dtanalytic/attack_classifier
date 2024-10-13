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
    
    conf = YAML().load(open(config_path))

    mlb = joblib.load(conf['prep_text']['mlb_fn'])
    data = pd.read_csv(conf['prep_text']['prep_fn'])
    data['labels'] = data['labels'].map(lambda x: eval(x))
    
    data, feat_data, vec = calc_select_feat_matr(data=data, target_col='labels', mlb=mlb, conf=conf)
    

    # joblib.dump(feat_data, conf['feat_gen']['feat_fn'])
    feat_data.to_csv(conf['feat_gen']['feat_fn'], index=False)
    data.drop(columns=['ttp', 'target_ttp']).to_csv(conf['feat_gen']['data_fn'], index=False)
    joblib.dump(vec, conf['feat_gen']['vec_fn'])
    
    
if __name__=='__main__':

    main()