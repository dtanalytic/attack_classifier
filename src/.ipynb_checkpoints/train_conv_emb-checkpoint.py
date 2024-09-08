from sklearn.metrics import average_precision_score, log_loss

from ruamel.yaml import YAML
import pandas as pd
import numpy as np
import joblib

from collections import defaultdict
from itertools import chain


import click
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR, MultiStepLR


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


import sys
sys.path.append('.')
from src.funcs import set_seed, TextVectorizer
from src.train_eval_model import metric_multi

class TextConv(nn.Module):

    def __init__(self, vocab_size, seq_len, embedding_dim, k_size, conv_num, drop_ratio, out_feat):
        super().__init__()
        seq_len = 512 if not seq_len else seq_len
        self.embed = nn.Embedding(num_embeddings=vocab_size+2, embedding_dim=embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=conv_num, kernel_size=k_size)
        self.dropout = nn.Dropout(p=drop_ratio)
        self.linear = nn.Linear(in_features = conv_num, out_features=out_feat)

    def forward(self, X, return_emb=False):
        # import pdb;pdb.set_trace()
        out = self.embed(X).permute(0,2,1)
        out = self.conv(out)
        out = F.relu(out)
        # размерность для усреднения
        s = out.size(dim=2)
        out = F.avg_pool1d(out,s).squeeze(dim=2)
        if return_emb:
            return out
        out = self.dropout(out)
        out = self.linear(out)

        return out


class TextDataset(torch.utils.data.Dataset):

    def __init__(self, X, y, vec_type, maxlen, vec, mlb):
        self.X = X
        self.y = y
        self.vec_type = vec_type
        self.vec = vec
        self.maxlen = maxlen
        self.mlb = mlb
        
    def __getitem__(self, idx):

        return torch.tensor(self.vec.vectorize(self.X.iloc[idx], vec_type=self.vec_type, maxlen=self.maxlen), dtype=torch.int64),\
                torch.tensor(self.mlb.transform(self.y.iloc[[idx]]).squeeze(0), dtype=torch.float32)

    def __len__(self):
        return self.y.shape[0]
        
@click.command()
@click.argument('config_path_dop')
@click.argument('config_path_main')
def main(config_path_dop, config_path_main):
    
    conf_dop = YAML().load(open(config_path_dop))
    conf_main = YAML().load(open(config_path_main))
    
    set_seed(conf_main['seed'])
    
    mlb = joblib.load(conf_main['prep_text']['mlb_fn'])
    data = pd.read_csv(conf_main['feat_gen']['data_fn'])
    
    data['target'] = data['target'].map(lambda x: eval(x))
    data['labels'] = data['labels'].map(lambda x: eval(x))
    
    
    
    tr_idx = data.query('split=="tr"').index
    val_idx = data.query('split=="val"').index
    ts_idx = data.query('split=="ts"').index


    maxlen = conf_dop['nn']['maxlen']
    batch_size = conf_dop['nn']['batch_size']
    vocab_size = conf_dop['nn']['vocab_size']

    embedding_dim = conf_dop['nn_conv']['embedding_dim']
    k_size = conf_dop['nn_conv']['k_size']
    conv_num = conf_dop['nn_conv']['conv_num']
    drop_ratio = conf_dop['nn_conv']['drop_ratio']
    

    epoch_num = conf_dop['nn']['epoch_num']
    learning_rate = conf_dop['nn']['learning_rate']
    l2 = conf_dop['nn']['l2']

    exp_gamma = conf_dop['nn']['exp_gamma']
    milestone_gamma = conf_dop['nn']['milestone_gamma']
    
    milestone_l = conf_dop['nn']['milestone_l']


    vec = TextVectorizer.from_dataframe(df=data.query('split=="tr"'), feat_col='prep_text', target_col=None, 
                                    feat_vocab_size=vocab_size, lang='english', preprocess=False)


    tr_ds = TextDataset(data.query('split=="tr"')['prep_text'], data.query('split=="tr"')['labels'], vec_type='digits', maxlen = maxlen, 
                        vec = vec, mlb = mlb)
    val_ds = TextDataset(data.query('split=="val"')['prep_text'], data.query('split=="val"')['labels'], vec_type='digits', maxlen = maxlen, 
                         vec = vec, mlb = mlb)
    ts_ds = TextDataset(data.query('split=="ts"')['prep_text'], data.query('split=="ts"')['labels'], vec_type='digits', maxlen = maxlen, 
                        vec = vec, mlb = mlb)
    
    
    tr_ld = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_ld = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)
    ts_ld = torch.utils.data.DataLoader(ts_ds, batch_size=batch_size)

    conv_net = TextConv(vocab_size=vocab_size, seq_len=maxlen, embedding_dim=embedding_dim,  
                        k_size=k_size, conv_num=conv_num, drop_ratio=drop_ratio, out_feat=len(mlb.classes_))

    val_iter_num = 1
    refresh_cache_iter = 10

    loss_fn = torch.nn.BCEWithLogitsLoss()
    conv_net = conv_net.to(DEVICE)
    optimizer = torch.optim.Adam(params = conv_net.parameters(), lr=learning_rate, weight_decay=l2)

    
    scheduler1 = ExponentialLR(optimizer, gamma=exp_gamma)
    scheduler2 = MultiStepLR(optimizer, milestones=milestone_l, gamma=milestone_gamma)
    
    loss_d = {}
    
    for epoch in range(1, epoch_num+1):
        loss_tr_l = []
        res_d = defaultdict(list)
        conv_net.train()
        tr_batch_num = len(tr_ld)
        tr_loss_epoch = 0
        for batch_tr in tr_ld:
            out = conv_net(batch_tr[0].to(DEVICE))
            
            optimizer.zero_grad()
    
            loss = loss_fn(out, batch_tr[1].to(DEVICE))
            
            loss.backward()
            optimizer.step()
        
            tr_loss_epoch = tr_loss_epoch + loss.item()
            with torch.no_grad():
                conv_net.eval()
                res_d['tr_target'].append(batch_tr[1].numpy()) 
                res_d['tr_pred'].append(out.sigmoid().cpu().numpy()) 
        
        res_d['tr_target'] = list(chain(*res_d['tr_target']))
        res_d['tr_pred'] = list(chain(*res_d['tr_pred']))
        
        scheduler1.step()
        scheduler2.step()
        
        if epoch%val_iter_num==0:
            conv_net.eval()
            val_batch_num = len(val_ld)
            val_loss_epoch = 0
            pr_auc = 0
            
            with torch.no_grad():
                for batch_val in val_ld:
                    pred = conv_net(batch_val[0].to(DEVICE))
                    val_loss = loss_fn(pred, batch_val[1].to(DEVICE))
                    val_loss_epoch = val_loss_epoch+val_loss.item()
                    sigm_preds = pred.sigmoid().cpu()
    
                    pr_auc = pr_auc + metric_multi(batch_val[1].numpy(), sigm_preds.numpy(), average_precision_score)[0]
    
                    res_d['target'].append(batch_val[1].numpy())
                    res_d['pred'].append(sigm_preds.numpy())
    
                res_d['target'] = list(chain(*res_d['target']))
                res_d['pred'] = list(chain(*res_d['pred']))
    
        loss_d[epoch] = {'log_loss_tr_batch':tr_loss_epoch/tr_batch_num,
                          'log_loss_val_batch':val_loss_epoch/val_batch_num,
                        'pr_auc_batch':pr_auc/val_batch_num,
                         'log_loss_val':metric_multi(np.array(res_d['target']), np.array(res_d['pred']), log_loss)[0],
                        'log_loss_tr':metric_multi(np.array(res_d['tr_target']), np.array(res_d['tr_pred']), log_loss)[0],
                         'pr_auc_val':metric_multi(np.array(res_d['target']), np.array(res_d['pred']), average_precision_score)[0],
                        'pr_auc_tr':metric_multi(np.array(res_d['tr_target']), np.array(res_d['tr_pred']), average_precision_score)[0]}

    with open(conf_dop['nn_conv']['metrics_dop_fn'], 'wt') as f_wr:
        json.dump(loss_d, f_wr)

    with open(conf_dop['nn_conv']['metrics_fn'], 'wt') as f_wr:
        json.dump({'max_pr_auc_tr':max([it['pr_auc_tr'] for it in loss_d.values()]),
'max_pr_auc_val':max([it['pr_auc_val'] for it in loss_d.values()])}, f_wr)


    ds = TextDataset(data['prep_text'], data['labels'], vec_type='digits', maxlen = maxlen, vec = vec, mlb = mlb)
    ld = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    emb_l = []
    
    with torch.no_grad():
        for batch in ld:
            emb = conv_net.forward(batch[0].to(DEVICE), return_emb=True)
        
            emb_l.append(emb.cpu().numpy())
    
    emb_l = list(chain(*emb_l))
    emb_l = [it.tolist() for it in emb_l]

    pd.DataFrame(emb_l).to_csv(conf_dop['nn_conv']['conv_emb_feat_fn'], index=False)

if __name__=='__main__':

    main()