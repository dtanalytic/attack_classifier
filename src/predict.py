import numpy as np
import pandas as pd

from transformers import BertTokenizer
from transformers import DataCollatorWithPadding
from transformers import RobertaTokenizer, RobertaModel

from torch.utils.data import DataLoader, Dataset
import torch

from ruamel.yaml import YAML
import joblib

from src.funcs import get_preds
from src.spec_nn_funcs import TextDFDataset, TextModelClass

def predict(pred_l):
    '''
    Parameters
    ----------
    pred_l - list of strings
    Returns
    -------
    pandas DataFrame with predicted tuples of:
                                            - taktics ('pred_str_tak' column)
                                            - technik ('pred_str_tech' column)
    
    '''
    conf = YAML().load(open('params.yaml'))
    conf_bert = YAML().load(open('dvc_pipes/bert/params_bert.yaml'))
    conf_bert_ttp = YAML().load(open('dvc_pipes/bert_ttp/params_bert_ttp.yaml'))
    conf_bert_ttp['nn'] = conf_bert_ttp['nn_ttp']
    conf_bert_ttp['nn_bert'] = conf_bert_ttp['nn_bert_ttp']

    mlb = joblib.load(conf['prep_text']['mlb_fn'])
    mlb_ttp = joblib.load(conf['prep_text']['ttp_mlb_fn'])

    model_bert_tak = torch.load(conf['train_fin']['model_taktic_fn'])
    model_bert_tech = torch.load(conf['train_fin']['model_technik_fn'])
    thresh_tech_l = joblib.load(conf['train_fin']['thresh_ttp_fn'])
    thresh_tak_l = joblib.load(conf['train_fin']['thresh_fn'])

    pred_df = predict_bert(pd.DataFrame({'sentence':pred_l}).assign(target=1), model_bert_tak, thresh_tak_l, conf_bert, suf='tak')

    pred_df = predict_bert(pred_df, model_bert_tech, thresh_tech_l, conf_bert_ttp, suf='tech')

    pred_df['pred_str_tech'] = pred_df['pred_tech'].map(lambda x: mlb_ttp.inverse_transform(np.array([x]))[0])
    pred_df['pred_str_tak'] = pred_df['pred_tak'].map(lambda x: mlb.inverse_transform(np.array([x]))[0])

    return pred_df
    
def predict_bert(pred_df, model, thresh_l, conf_bert, suf):
    
    batch_size = conf_bert['nn']['batch_size']
    
    bert_type = conf_bert['nn_bert']['bert_type']
    
    MAX_SEQ_LENGTH = conf_bert['nn']['maxlen']
    
    # model_bert
    
    if bert_type == 'secbert_plus':
        
        checkpoint = 'data/external/models/SecureBERT_Plus/snapshots/4c48ccdb8d2019f179b07dfa27656c655394d78e'
        tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
        tokenizer_opts = {'max_length':MAX_SEQ_LENGTH, 'return_tensors':"pt", 'padding':True, 'truncation':True, 'add_special_tokens':True}
        
    elif bert_type == 'secbert':
        checkpoint = 'data/external/models/SecBERT/snapshots/7c603df5bc4c5ba9c731bcc2ea0ab2db36e104cb'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer_opts = {'max_length':MAX_SEQ_LENGTH, 'return_tensors':"pt", 'padding':True, 'truncation':True, 'add_special_tokens':True}
    
    elif bert_type == 'scibert':
        # checkpoint = 'allenai/scibert_scivocab_uncased'
        checkpoint = 'data/external/models/scibert_scivocab_uncased/snapshots/24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1'
        tokenizer = BertTokenizer.from_pretrained(checkpoint, max_length=512)
        tokenizer_opts = {'return_tensors':"pt", 'truncation':True,
                          'padding':'max_length', 'max_length':MAX_SEQ_LENGTH}
    
    
    
    ds = TextDFDataset(pred_df, tokenizer=tokenizer, tokenizer_opts=tokenizer_opts)
    ld = DataLoader(ds, batch_size = batch_size, shuffle = False, collate_fn = DataCollatorWithPadding(tokenizer=tokenizer))
    

    Y_pred_proba = np.array(get_preds(model, ld=ld)['pred'])

    pred_df[f'proba_{suf}'] = Y_pred_proba.tolist()
    pred_df[f'pred_{suf}'] = pred_df[f'proba_{suf}'].map(lambda x: [int(val>=thresh) for val, thresh in zip(x, thresh_l)])
    return pred_df