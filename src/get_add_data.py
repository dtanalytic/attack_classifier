import pandas as pd
import numpy as np
import joblib
import json
from itertools import chain
import click
import os

from ruamel.yaml import YAML

import sys
sys.path.append('.')

@click.command()
@click.argument('config_path')
def main(config_path):
    
    conf = YAML().load(open(config_path))
    mitre_attack_df = pd.read_csv(conf['get_data']['data_mitre_attack_proc_fn'])
    mitre_attack_df = mitre_attack_df.assign(labels = mitre_attack_df['labels'].map(lambda x:[x]))

    # mitre_attack_df = mitre_attack_df[mitre_attack_df['sentence'].str.split().str.len()>=10].reset_index(drop=True)
    # write same sentences for different technick 
    mitre_attack_df[(mitre_attack_df.duplicated('sentence', keep=False))
            &(mitre_attack_df['sentence']!=' ')\
            &(mitre_attack_df['sentence'].str.split().str.len()>=10)\
            &(~mitre_attack_df['sentence'].str.contains('*', regex=False))].sort_values(by='sentence')\
            .to_csv('data/artifacts/duple_text_dif_techniks.csv')
    
    # mitre_attack_df = mitre_attack_df.drop_duplicates('sentence')
    
    with open(conf['get_data']['label2tactic_fn'], 'rt') as f_r:
        label2tactic = json.load(f_r)

    tram_df = pd.read_json(conf['get_data']['tram_fn']).drop(columns='doc_title')
    sel = tram_df.sentence.str.findall(';').str.len()>0
    tram_df[sel].to_csv('data/artifacts/tram_rubbish.csv', index=False)
    tram_df = tram_df[~sel]
    tram_df = tram_df.drop_duplicates(subset='sentence')
    
    

    
    if conf['get_data']['use_alt_mitre']:
        par_d = mitre_attack_df.explode('labels').set_index('labels')['par_name'].to_dict()
        mitr_alt_data = pd.read_csv(conf['get_data']['mitre_alt_fn'])
        mitr_alt_data['labels'] = mitr_alt_data[['label_subtec', 'label_tec']].apply(lambda x: list(set((x[0], x[1]))), axis=1)
        mitr_alt_data = mitr_alt_data[['sentence', 'labels']]
        mitr_alt_data['par_name'] = mitr_alt_data['labels'].map(lambda x: sorted([par_d[it] if it in par_d and len(it)>6 else '' for it in x])[-1])

        mitr_df = mitr_alt_data

    else:
        mitr_df = mitre_attack_df

    if conf['get_data']['use_tram_f']:
        df = pd.concat([mitr_df, tram_df], ignore_index=True)
    else:
        df = mitr_df
        
    if conf['get_data']['use_reports_f']:
        DN = conf['get_data']['rep_dn']
        fns = [f'{DN}/{it}' for it in os.listdir(DN) if 'json' in it]
        
        tab_df = pd.concat([pd.read_json(fn).explode('tables') for fn in fns], ignore_index=True)
        tab_df['tables'] = tab_df['tables'].map(lambda x: [(it1, it2) 
                                        for it1, it2 in zip(x['TechniqueID'].values(), x['Procedure'].values())])
        tab_df = tab_df.explode('tables').drop_duplicates().reset_index(drop=True)
        tab_df['technic'] = tab_df['tables'].map(lambda x: x[0])
        tab_df['sentence'] = tab_df['tables'].map(lambda x: x[1])
        
        tab_df = tab_df[tab_df['sentence'].str.split().str.len()>3]
        tab_df['labels'] = tab_df['technic'].map(lambda x: [x.upper()])
        
        # тут и мобильные угрозы есть, поэтому фильтруем их
        tab_df = tab_df[tab_df['labels'].map(lambda x: all([it in label2tactic for it in x]))]

        df = pd.concat([df, tab_df[['sentence',	'labels', 'report_path']].rename(columns={'report_path':'url'})], ignore_index=True)
    
    df['origin_labels'] = df['labels']
    if conf['get_data']['target'] == 'tactic':          
      # df['labels'] = df['labels'].map(lambda x: list(chain(*[label2tactic[it] for it in x])))
      df['labels'] = df['labels'].map(lambda x: list(chain(*[label2tactic[it] if it in label2tactic else '' for it in x ])))

    df.sentence = df.sentence.str.strip()

    # еще есть скрытые, у которых отличия только в паре слов или символов
    df['shadow_duples'] = df.sentence.str[:40]+ df.sentence.str[50] + df.sentence.str[-40:]
    df[df['shadow_duples'].duplicated(keep=False)].sort_values(by='sentence').to_csv('data/artifacts/shadow_duples.csv', index=False)
    df = df.drop_duplicates('shadow_duples').drop(columns='shadow_duples')

    
    # при разбиении на абзацы описаний вылезают дубли
    df = df.drop_duplicates('sentence')

    # есть тексты очень малые и при разбиении на абзацы списки превращаются в мини перечисления
    df = df[df['sentence'].str.split().str.len()>=10].reset_index(drop=True)
    
    # joblib.dump(df, conf['get_data']['data_fn'])
    df.to_csv(conf['get_data']['data_fn'], index=False)
    
if __name__=='__main__':

    main()