import torch
from itertools import product
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    