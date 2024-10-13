import pandas as pd
import numpy as np
import joblib
import click

from ruamel.yaml import YAML

import torch
import transformers
from transformers import RobertaTokenizer, RobertaTokenizerFast

from sentence_transformers import SentenceTransformer

import sys
sys.path.append('.')

from src.spec_funcs import calc_select_feat_matr
from src.aug_sent import get_aug_sent_l, make_mask_list, filter_sent_distance



@click.command()
@click.argument('config_path')
def main(config_path):
    
    conf_ttp = YAML().load(open(config_path))
    conf = YAML().load(open('params.yaml'))
    conf_dop = YAML().load(open('dvc_pipes/bert_ttp/params_bert_ttp.yaml'))

    # чтобы новый конф работал вместо старого в функции
    conf_ttp['feat_gen'] = conf_ttp['feat_gen_ttp'] 
    conf_ttp['seed'] = conf['seed']
    conf_ttp['use_only_proc'] = conf['use_only_proc']
    
    mlb = joblib.load(conf['prep_text']['ttp_mlb_fn'])
    
    data = pd.read_csv(conf['prep_text']['prep_fn'])
    

    data['ttp'] = data['ttp'].map(lambda x: eval(x))

    if conf_ttp['feat_gen']['add_aug_sents']:


            
        data['sentence_source'] = data['sentence']

        num_mask_pos = conf_ttp['feat_gen']['aug_num_mask_pos']
        repeate_mask = conf_ttp['feat_gen']['aug_repeate_mask']
        topk = conf_ttp['feat_gen']['aug_topk']
        synth_thresh_class_num = conf_ttp['feat_gen']['aug_synth_thresh_class_num']
        split = conf_ttp['feat_gen']['aug_split']
        emb_path = conf_ttp['feat_gen']['aug_emb_path']
        checkpoint = conf_ttp['feat_gen']['aug_secbert_path']
        sim_thresh = conf_ttp['feat_gen']['aug_sim_thresh']
        
        tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint)
        tokenizer_opts = {'max_length':conf_dop['nn_ttp']['maxlen'], 'return_tensors':"pt", 
                          'padding':True, 'truncation':True, 'add_special_tokens':True}

        model = transformers.RobertaForMaskedLM.from_pretrained(checkpoint)
        smodel = SentenceTransformer(emb_path)

        
        mask_token = tokenizer.mask_token

        sel_dop = (data.split==split) if split else pd.Series([True]*len(data))
        
        mini_ttp_l = data[sel_dop].explode('ttp').groupby('ttp').size().loc[lambda x: x<=synth_thresh_class_num].index.tolist()
        mini_ttp_l = [it for it in mini_ttp_l if not 'rare'==it]

        sel = (data.ttp.map(lambda x: len(set(x).intersection(mini_ttp_l)) > 0)) & (sel_dop)
    

        data.loc[sel, 'sentence'] = data.loc[sel, 'sentence'].map(lambda x: make_mask_list(x, mask_token, 
                                                                                     num_mask_pos=num_mask_pos, repeate=repeate_mask))
        # import pdb;pdb.set_trace()
        data.loc[sel, 'sentence'] = data.loc[sel, 'sentence'].map(lambda x:  get_aug_sent_l( sents = x[1:], topk_words_num=topk, 
                              model=model, tokenizer=tokenizer, tokenizer_opts=tokenizer_opts) if len(x)>1 else [])

        size_dif = data.explode('sentence').shape[0] - data.shape[0]
        print(f'После добавления всей аугментации размер абс - {size_dif}, отн - {size_dif/data.shape[0]:.0%} ')

        data.loc[sel, 'sentence'] = data.loc[sel, ['sentence', 'sentence_source']].apply(lambda x: filter_sent_distance(x['sentence'], x['sentence_source'], smodel, sim_thresh)  if len(x['sentence'])>0 else [], axis=1)

        data.loc[~sel, 'sentence'] = data.loc[~sel, 'sentence_source']
        data.loc[sel, 'sentence'] = data.loc[sel, ['sentence', 'sentence_source']].apply(lambda x: x['sentence']+[x['sentence_source']], axis=1)

        size_dif = data.explode('sentence').shape[0] - data.shape[0]
        print(f'После добавления всей аугментации за вычетом предложений по threshhold размер абс - {size_dif}, отн - {size_dif/data.shape[0]:.0%} ')

        data = data.explode('sentence').drop_duplicates(subset=['sentence']).reset_index()
        
    # тут feat_gen_ttp ключ надо заменить на feat_gen и добавить в  with seed, use_only_proc
    data, feat_data, vec = calc_select_feat_matr(data=data, target_col='ttp', mlb=mlb, conf=conf_ttp)
    

    # joblib.dump(feat_data, conf['feat_gen']['feat_fn'])
    feat_data.to_csv(conf_ttp['feat_gen_ttp']['feat_fn'], index=False)
    data.to_csv(conf_ttp['feat_gen_ttp']['data_fn'], index=False)
    joblib.dump(vec, conf_ttp['feat_gen_ttp']['vec_fn'])
    
    
if __name__=='__main__':

    main()