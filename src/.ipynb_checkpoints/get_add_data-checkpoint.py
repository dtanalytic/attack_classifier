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
    mitre_attack_df = pd.read_csv(conf['get_data']['data_mitre_attack_proc_fn'])
    mitre_attack_df = mitre_attack_df.assign(labels = mitre_attack_df['labels'].map(lambda x:[x]))
    
    with open(conf['get_data']['label2tactic_fn'], 'rt') as f_r:
        label2tactic = json.load(f_r)

    tram_df = pd.read_json(conf['get_data']['tram_fn']).drop(columns='doc_title')

    mitr_alt_data = pd.read_csv(conf['get_data']['mitre_alt_fn'])
    
    mitr_alt_data['labels'] = mitr_alt_data[['label_subtec', 'label_tec']].apply(lambda x: list(set((x[0], x[1]))), axis=1)
    mitr_alt_data = mitr_alt_data[['sentence', 'labels']]

    if conf['get_data']['use_alt_mitre']:
      mitr_df = mitr_alt_data
    else:
      mitr_df = mitre_attack_df

    if conf['get_data']['use_tram_f']:
        df = pd.concat([mitr_df, tram_df], ignore_index=True)
    else:
        df = mitr_df
        
    # тут по хорошему получение отчетов
    if conf['get_data']['use_reports_f']:
        rep_data  = joblib.load('/content/drive/MyDrive/Colab Notebooks/texts/sec_bert/data/rep_data.pkl')
        df = pd.concat([df, rep_data[['sentence', 'labels']]], ignore_index=True)
    
    df['origin_labels'] = df['labels']
    if conf['get_data']['target'] == 'tactic':          
      df['labels'] = df['labels'].map(lambda x: list(chain(*[label2tactic[it] for it in x])))

    
    # joblib.dump(df, conf['get_data']['data_fn'])
    df.to_csv(conf['get_data']['data_fn'], index=False)
    
if __name__=='__main__':

    main()