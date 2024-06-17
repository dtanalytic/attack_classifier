import pandas as pd
import numpy as np
import joblib
from nltk.corpus import stopwords
import click
from sklearn.preprocessing import MultiLabelBinarizer
from pandarallel import pandarallel


from ruamel.yaml import YAML

import sys
sys.path.append('.')
from src.funcs import preprocess_text
from src.constants import CLASSES


pandarallel.initialize(progress_bar=True, nb_workers=0)

@click.command()
@click.argument('config_path')
def main(config_path):
    
    conf = YAML().load(open(config_path))


    mlb = MultiLabelBinarizer(classes=CLASSES)
    mlb.fit([[c] for c in CLASSES])
    joblib.dump(mlb, conf['prep_text']['mlb_fn'])

    if conf['prep_text']['stop_words']:
        stop_words_l = set(stopwords.words('english'))
    else:
        stop_words_l = []
    
    data = pd.read_json('data/multi_label.json').drop(columns='doc_title')

    if conf['prep_text']['labelled_text_only']:
        data = data[data['labels'].str.len()>1]

    data['prep_text'] = data['sentence'].apply(lambda x: preprocess_text(x, to_lower=conf['prep_text']['lower'], 
     min_len_token=conf['prep_text']['min_len_token'], max_len_token=conf['prep_text']['max_len_token'], stop_words_l=stop_words_l, 
     stem_lemm_flag=conf['prep_text']['stem_lemm'], save_text=True, save_only_words= conf['prep_text']['save_only_words'], 
        lang='english'))
    
    data.to_csv(conf['prep_text']['prep_fn'], index=False)
    

if __name__=='__main__':

    main()