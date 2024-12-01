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

from src.spec_funcs import load_mitr, prep_mitr
from src.constants import (regexp_email, regexp_cve, regexp_url, regexp_domain, regexp_registry,  regexp_fpath, 
                             regexp_fname, regexp_ipv4, regexp_ipv6,  regex_domain_zone,
                            regexp_hash_md5, regexp_hash_sha1, regexp_hash_sha256, regexp_hash_sha512, regexp_hash_ssdeep, 
                             regexp_coins_eth, regexp_coins_btc, regexp_coins_bch, regexp_coins_ltc,
                            regexp_coins_doge, regexp_coins_dash, regexp_coins_xmr, regexp_coins_neo, regexp_coins_xrp)
from src.spec_funcs import replace_entities


def train():
    '''
    trains 2 models
    reads configs from param.yaml, dvc_pipes/ttp/params_ttp.aml, dvc_pipes/bert_ttp/params_ttp.aml
    '''
    # получаем данные митра
    conf = YAML().load(open('params.yaml'))

    load_external_data(conf)


    
def load_external_data(conf):

    mitre_df, main_descr_df, proc_df = load_mitr(conf['get_data']['mitre_attack_fn'])
    mitre_attack_df = prep_mitr(main_descr_df, proc_df, conf)
    

    label2tactic = mitre_attack_df.set_index('labels')['kill_chain_tags'].to_dict()
    
    with open(conf['get_data']['label2tactic_fn'], 'wt') as f_wr:
        json.dump(label2tactic, f_wr)

        # ------------------------
    # сохранение ВСЕ ЛИ НАДО???
    mitre_attack_df = mitre_attack_df[['sentence', 'labels', 'url', 'par_name', 'is_proc']]
    mitre_attack_df.to_csv(conf['get_data']['data_mitre_attack_proc_fn'], index=False)
    
    mitre_df.to_csv(conf['get_data']['data_mitre_fn'], index=False)

    mitre_attack_df = mitre_attack_df.assign(labels = mitre_attack_df['labels'].map(lambda x:[x]))

    
            
    
    tram_df = pd.read_json(conf['get_data']['tram_fn']).drop(columns='doc_title')
    sel = tram_df.sentence.str.findall(';').str.len()>0

    tram_df = tram_df[~sel]
    tram_df = tram_df.drop_duplicates(subset='sentence')

    mitr_df = mitre_attack_df
    
    df = pd.concat([mitr_df, tram_df], ignore_index=True)
    

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
    df['origin_ttp'] = df['labels']

    df['labels'] = df['labels'].map(lambda x: list(chain(*[label2tactic[it] if it in label2tactic else '' for it in x ])))

    df.sentence = df.sentence.str.strip()

    # еще есть скрытые, у которых отличия только в паре слов или символов
    df['shadow_duples'] = df.sentence.str[:40]+ df.sentence.str[50] + df.sentence.str[-40:]
    df[df['shadow_duples'].duplicated(keep=False)].sort_values(by='sentence').to_csv('data/artifacts/shadow_duples.csv', index=False)
    df = df.drop_duplicates('shadow_duples').drop(columns='shadow_duples')

    # при разбиении на абзацы описаний вылезают дубли
    df = df.drop_duplicates('sentence')

    # есть тексты очень малые и при разбиении на абзацы списки превращаются в мини перечисления
    df = df[df['sentence'].str.split().str.len()>=5].reset_index(drop=True)
    
    # NEED?
    df.to_csv(conf['get_data']['data_fn'], index=False)

    # таргет execution-а добавляем
    tech_l = ['T1218', 'T1480', 'T1202', 'T1216', 'T1127']
    sel = (df['par_name'].fillna('').str.contains('Execution')) & \
        (df['origin_labels'].astype(str).map(lambda x: len([it for it in tech_l if it in x])>0))
        
    df.loc[sel, 'labels'] = df.loc[sel, 'labels'].map(lambda x: x + ['execution'])

    # NEED?
    df.to_csv(conf['get_data']['data_filt_fn'], index=False)

    # --------------------------
    # prep.py
    data = df
    
    if conf['prep_text']['replace_entities']:
        # начиная с python 3.7 порядок ключей сохраняется, поэтому можно не упорядочивать
        pat_d = {it:globals()[it] for it in globals() if 'regexp_' in it}
        data['sentence'] = data['sentence'].map(lambda x: replace_entities(x, pat_d))

    
    # другие названия в квадратных скобках убираем
    data['sentence'] = data['sentence'].str.replace(r'\[(\w+)\]', r'\1', regex=True)


    # после того, как убрал названия в круглых скобках вылезли дубли, например, в отчете та же формулировка, как
    # и в первоисточнике на митр только уже без круглых скобок
    data = data.drop_duplicates(subset=['sentence']).reset_index(drop=True)
    
    symb_l = ['\xe4', '\u202f', '\u2192']
    for symb in symb_l:
        data['sentence'] = data['sentence'].str.replace(symb, '')

    
    # шо делать с ттп
    ttp_counts_thresh = conf['prep_text']['ttp_counts_thresh']

