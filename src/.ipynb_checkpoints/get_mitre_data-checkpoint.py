import pandas as pd
import numpy as np
import joblib
import json
import re
from itertools import chain
import click

from ruamel.yaml import YAML

import sys
sys.path.append('.')
from src.spec_funcs import load_mitr, prep_mitr

@click.command()
@click.argument('config_path')
def main(config_path):
    
    conf = YAML().load(open(config_path))

    mitre_df, main_descr_df, proc_df = load_mitr(conf['get_data']['mitre_attack_fn'])
    mitre_attack_df = prep_mitr(main_descr_df, proc_df, conf)
    
    # ------------------------
    # сохранение
    mitre_attack_df[['sentence', 'labels', 'url', 'par_name', 'is_proc']].to_csv(conf['get_data']['data_mitre_attack_proc_fn'], index=False)
    
    with open(conf['get_data']['label2tactic_fn'], 'wt') as f_wr:
        json.dump(mitre_attack_df.set_index('labels')['kill_chain_tags'].to_dict(), f_wr)

    mitre_df.to_csv(conf['get_data']['data_mitre_fn'], index=False)
    
    
if __name__=='__main__':

    main()