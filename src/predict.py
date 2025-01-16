import numpy as np

from transformers import BertTokenizer
from transformers import DataCollatorWithPadding
from transformers import RobertaTokenizer, RobertaModel

from torch.utils.data import DataLoader, Dataset

from src.funcs import get_preds
from src.spec_nn_funcs import TextDFDataset, TextModelClass

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