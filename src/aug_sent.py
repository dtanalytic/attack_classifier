import torch
from itertools import product
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ruamel.yaml import YAML
import torch
import transformers
from transformers import RobertaTokenizer, RobertaTokenizerFast
from sentence_transformers import SentenceTransformer


import sys
sys.path.append('.')

from src.funcs import set_seed

conf_seed = YAML().load(open('params.yaml'))
set_seed(conf_seed['seed'])


def make_mask_list(sent, mask_token, num_mask_pos, repeate=1):
    res_l = []
    w_l = sent.split()
    for _ in range(repeate):
        for num in num_mask_pos:
            w_l_c = w_l.copy()
            ids_big_words_l = [id for id, it in enumerate(w_l_c) if len(it) > 4]
            if len(ids_big_words_l)>=num:     
                idx_ar = np.random.choice(range(len(ids_big_words_l)), size=num, replace=False)
                for ind in idx_ar:
                    w_l_c[ids_big_words_l[ind]] = mask_token
                res_l.append(' '.join(w_l_c))
    return [sent]+res_l


def update_sent_pair(sent, pair, mask_token):  
    sent_w_l = sent.split()
    updatesent_w_l = []
    counter_mask = 0
    for i, w in enumerate(sent_w_l):
        if w==mask_token:
            # если маска дальше максимальной длины токенизатора, то она не будет предсказана и вставляем спец символ
            updatesent_w_l.append(pair[counter_mask]) if len(pair) > counter_mask else updatesent_w_l.append('<mask_out_of_bounds>')
            counter_mask +=1
        else:
            updatesent_w_l.append(w)
        
    return ' '.join(updatesent_w_l)


def get_aug_sent_l(sents, topk_words_num, model, tokenizer, tokenizer_opts):
    '''
    Возвращает примеры с видами предложений, где для каждой вариации маски возвращается topk_words_num*mask_num
    Количество вариаций масок = len(sents) = len(num_mask_pos)*repeate_mask
    Количество итоговых комбинаций может не совпадать так как у токенизатора secure_bert_plus есть одинаковые символы, отличающиеся на пробел (например, 43, 4839)
    '''
    # np.where((token_ids.squeeze() == tokenizer.mask_token_id))
    token_ids = tokenizer(sents, **tokenizer_opts)['input_ids']
    masked_position = (token_ids == tokenizer.mask_token_id).nonzero()
    with torch.no_grad():
        output = model(token_ids)
    # output.logits is same as output[0]
    last_hidden_state = output.logits

    alias_sents = []
    
    for sent_num, sent in enumerate(sents):
        # torch.where
        masked_pos = masked_position[np.where(masked_position[:,0]==sent_num),1]
        mask_hidden_state = last_hidden_state[sent_num][masked_pos]     
        idx_t = torch.topk(mask_hidden_state, k=topk_words_num, dim=2)[1].squeeze(0)
        w_l = []
        for i in range(idx_t.shape[0]):
            w_l.append([tokenizer.decode(j.item()).strip() for j in idx_t[i]])
        w_tuples_l = list(set(product(*w_l)))
        alias_sents.extend([update_sent_pair(sent, pair, tokenizer.mask_token) for pair in w_tuples_l])
    return alias_sents



def filter_sent_distance(sents, sent_source, smodel, sim_thresh):
    embeddings = smodel.encode(sents+[sent_source])
    sims = cosine_similarity([embeddings[-1]], embeddings[:-1])[0]

    return [it for i, it in enumerate(sents) if sims[i]>sim_thresh]



def add_aug_sents(data, conf_ttp, max_len):
     
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
    tokenizer_opts = {'max_length':max_len, 'return_tensors':"pt", 
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

    return data

    