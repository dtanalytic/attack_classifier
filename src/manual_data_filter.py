import pandas as pd
import numpy as np
import joblib
import json
from itertools import chain
import click

from ruamel.yaml import YAML

import sys
sys.path.append('.')

@click.command()
@click.argument('config_path')
def main(config_path):
    
    conf = YAML().load(open(config_path))
    
    df = pd.read_csv(conf['get_data']['data_fn'])
    
    df['labels'] = df['labels'].map(lambda x: eval(x))
    df['origin_labels'] = df['origin_labels'].map(lambda x: eval(x))

    tech_l = ['T1218', 'T1480', 'T1202', 'T1216', 'T1127']
    sel = (df['par_name'].fillna('').str.contains('Execution')) & \
        (df['origin_labels'].astype(str).map(lambda x: len([it for it in tech_l if it in x])>0))
        
    df.loc[sel, 'labels'] = df.loc[sel, 'labels'].map(lambda x: x + ['execution'])
        

    df.to_csv(conf['get_data']['data_filt_fn'], index=False)
    
if __name__=='__main__':

    main()