import pandas as pd
import numpy as np
import joblib
import json
from itertools import chain
import click
import os

from ruamel.yaml import YAML

from sklearn.preprocessing import MultiLabelBinarizer
import torch



import sys
sys.path.append('.')
from src.funcs import set_seed

conf_seed = YAML().load(open('params.yaml'))
set_seed(conf_seed['seed'])

from src.spec_funcs import load_mitr, prep_mitr
from src.constants import (regexp_email, regexp_cve, regexp_url, regexp_domain, regexp_registry,  regexp_fpath, 
                             regexp_fname, regexp_ipv4, regexp_ipv6,  regex_domain_zone,
                            regexp_hash_md5, regexp_hash_sha1, regexp_hash_sha256, regexp_hash_sha512, regexp_hash_ssdeep, 
                             regexp_coins_eth, regexp_coins_btc, regexp_coins_bch, regexp_coins_ltc,
                            regexp_coins_doge, regexp_coins_dash, regexp_coins_xmr, regexp_coins_neo, regexp_coins_xrp)
from src.spec_funcs import replace_entities


from src.spec_nn_funcs import train_bert
from src.aug_sent import add_aug_sents
from src.spec_nn_funcs import TextDFDataset, TextModelClass, train_eval_bert


def train():
    '''
    trains 2 models
    reads configs from param.yaml, dvc_pipes/ttp/params_ttp.aml, dvc_pipes/bert_ttp/params_ttp.aml
    '''
    # получаем данные митра

    conf = YAML().load(open('params.yaml'))
    conf_bert = YAML().load(open('dvc_pipes/bert/params_bert.yaml'))
    conf_bert_ttp = YAML().load(open('dvc_pipes/bert_ttp/params_bert_ttp.yaml'))

    df = load_external_data(conf)
    df, mlb, mlb_ttp = enc_classes(df, conf, use_rare_ttp=False)

    # на самом деле 208 train тут уже есть - синтетика
    df['split'] = df['split'].fillna('tr')

    model_bert, loss_bert_d, thresh_l, _, _, _, (p_tr_micro, r_tr_micro, f1_tr_micro, p_tr_macro, r_tr_macro, f1_tr_macro) = train_bert(df, mlb, conf, conf_bert, target_col= 'labels', thresh_space_l=[])


    conf_ttp = YAML().load(open('dvc_pipes/ttp/params_ttp.yaml'))
    # NEED ?
    conf = YAML().load(open('params.yaml'))
    
    # чтобы новый конф работал вместо старого в функции
    conf_ttp['feat_gen'] = conf_ttp['feat_gen_ttp'] 
    conf_ttp['seed'] = conf['seed']
    conf_ttp['use_only_proc'] = conf['use_only_proc']
    
    mlb_ttp = joblib.load(conf['prep_text']['ttp_mlb_fn'])

    if conf_ttp['feat_gen']['add_aug_sents']:
        df_ttp = add_aug_sents(df.copy(), conf_ttp, conf_bert_ttp['nn_ttp']['maxlen'])
    else:
        df_ttp = df.copy()
    
    df_ttp['target'] = mlb_ttp.transform(df_ttp['ttp']).tolist()

    # NEED ?
    conf = YAML().load(open('params.yaml'))
    
    conf['feat_gen'] = conf_ttp['feat_gen_ttp']
    conf['train_eval_model'] = conf_ttp['train_eval_model_ttp']
    
    
    conf_bert_ttp['nn'] = conf_bert_ttp['nn_ttp']
    conf_bert_ttp['nn_bert'] = conf_bert_ttp['nn_bert_ttp']
    
    
    model_bert_ttp, loss_bert_ttp_d, thresh_ttp_l, _, _, _, (p_tr_micro, r_tr_micro, f1_tr_micro, p_tr_macro, r_tr_macro, f1_tr_macro) = train_bert(df_ttp, mlb_ttp, conf, conf_bert_ttp, target_col= 'ttp', thresh_space_l=np.arange(0.001, 1, 0.002))

    torch.save(model_bert_ttp, conf['train_fin']['model_technik_fn'])
    torch.save(model_bert, conf['train_fin']['model_taktic_fn'])
    
    joblib.dump(thresh_ttp_l, conf['train_fin']['thresh_ttp_fn'])
    joblib.dump(thresh_l, conf['train_fin']['thresh_fn'])
    
def load_external_data(conf):

    mitre_df, main_descr_df, proc_df = load_mitr(conf['get_data']['mitre_attack_fn'])
    mitre_attack_df = prep_mitr(main_descr_df, proc_df, conf)
    
    label2tactic = mitre_attack_df.set_index('labels')['kill_chain_tags'].to_dict()
    
    # with open(conf['get_data']['label2tactic_fn'], 'wt') as f_wr:
    #     json.dump(label2tactic, f_wr)

    # ------------------------
    # NEED?
    mitre_attack_df = mitre_attack_df[['sentence', 'labels', 'url', 'par_name', 'is_proc']]
    # mitre_attack_df.to_csv(conf['get_data']['data_mitre_attack_proc_fn'], index=False)
    
    # mitre_df.to_csv(conf['get_data']['data_mitre_fn'], index=False)

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
    if conf['get_data']['ignore_subt']:
        df['origin_labels'] = df['origin_labels'].map(lambda x: [it.split('.')[0] for it in x])
            
    df['labels'] = df['labels'].map(lambda x: list(chain(*[label2tactic[it] if it in label2tactic else '' for it in x ])))

    df.sentence = df.sentence.str.strip()

    # еще есть скрытые, у которых отличия только в паре слов или символов
    df['shadow_duples'] = df.sentence.str[:40]+ df.sentence.str[50] + df.sentence.str[-40:]
    df = df.drop_duplicates('shadow_duples').drop(columns='shadow_duples')

    # при разбиении на абзацы описаний вылезают дубли
    df = df.drop_duplicates('sentence')

    # есть тексты очень малые и при разбиении на абзацы списки превращаются в мини перечисления
    df = df[df['sentence'].str.split().str.len()>=5].reset_index(drop=True)
    
    # NEED?
    # df.to_csv(conf['get_data']['data_fn'], index=False)

    # таргет execution-а добавляем
    tech_l = ['T1218', 'T1480', 'T1202', 'T1216', 'T1127']
    sel = (df['par_name'].fillna('').str.contains('Execution')) & \
        (df['origin_labels'].astype(str).map(lambda x: len([it for it in tech_l if it in x])>0))
        
    df.loc[sel, 'labels'] = df.loc[sel, 'labels'].map(lambda x: x + ['execution'])

    # NEED?
    # df.to_csv(conf['get_data']['data_filt_fn'], index=False)

    # --------------------------
    # prep.py
    
    if conf['prep_text']['replace_entities']:
        # начиная с python 3.7 порядок ключей сохраняется, поэтому можно не упорядочивать
        pat_d = {it:globals()[it] for it in globals() if 'regexp_' in it}
        df['sentence'] = df['sentence'].map(lambda x: replace_entities(x, pat_d))

    
    # другие названия в квадратных скобках убираем
    df['sentence'] = df['sentence'].str.replace(r'\[(\w+)\]', r'\1', regex=True)


    # после того, как убрал названия в круглых скобках вылезли дубли, например, в отчете та же формулировка, как
    # и в первоисточнике на митр только уже без круглых скобок
    df = df.drop_duplicates(subset=['sentence']).reset_index(drop=True)
    
    symb_l = ['\xe4', '\u202f', '\u2192']
    for symb in symb_l:
        df['sentence'] = df['sentence'].str.replace(symb, '')

    if conf['prep_text']['include_chatgpt_aug']:
        DN = conf['prep_text']['chatgpt_dn']   
        synth_df = pd.concat([pd.read_csv(f'{DN}/{it}') for it in os.listdir(DN) if not '.ipynb_checkpoints' in it], ignore_index=True)
        synth_df['labels'] = synth_df['labels'].map(lambda x: eval(x))
        synth_df['origin_labels'] = synth_df['origin_labels'].map(lambda x: eval(x))
        synth_df['origin_ttp'] = synth_df['origin_ttp'].map(lambda x: eval(x))
        df = pd.concat([df, synth_df], ignore_index=True)    
    
    
    return df

def enc_classes(df, conf, use_rare_ttp):

    mlb_ttp = MultiLabelBinarizer()
    
    if use_rare_ttp:
        ttp_counts_thresh = conf['prep_text']['ttp_counts_thresh']
        ttp_l = df['origin_labels'].explode('origin_labels').value_counts().loc[lambda x: x>ttp_counts_thresh].index.tolist()    
        mlb_ttp.fit([[c] for c in ttp_l+['rare']])
        df['ttp'] = df['origin_labels'].map(lambda x: [it if it in ttp_l else 'rare' for it in x] )
    else:
        ttp_l = df['origin_labels'].explode('origin_labels').value_counts().index.tolist()    
        mlb_ttp.fit([[c] for c in ttp_l])
        df['ttp'] = df['origin_labels']

    # maybe need to rename target before bert_ttp prediction, because uses same column
    # data['target'] = mlb.transform(data[target_col]).tolist()
    df['target_ttp'] = mlb_ttp.transform(df['ttp']).tolist()


    CLASSES = df.explode('labels')['labels'].dropna().unique()
    mlb = MultiLabelBinarizer(classes=CLASSES)
    mlb.fit([[c] for c in CLASSES])


    df['target'] = mlb.transform(df['labels']).tolist()


    if not use_rare_ttp:
        joblib.dump(mlb, conf['prep_text']['mlb_fn'])
        joblib.dump(mlb_ttp, conf['prep_text']['ttp_mlb_fn'])
        
    return df, mlb, mlb_ttp

