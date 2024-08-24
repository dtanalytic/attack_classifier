import re
import random
import numpy as np

import nltk
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords

import string
from string import punctuation

from collections import Counter

from pymystem3 import Mystem


import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



class Vocabulary(object):

    def __init__(self, add_unk=True, add_mask=True):

        self.token2idx = {}
        self.idx2token = {}

        if add_mask:
          self.add_token('MASK')

        if add_unk:
          self.add_token('UNK')

    def add_token(self, token):

        if token not in self.token2idx:
            index = len(self.token2idx)
            self.token2idx[token] = index
            self.idx2token[index] = token

    def lookup_token(self, token):

        return self.token2idx.get(token, 1)

    def lookup_index(self, index):

        if index not in self.idx2token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self.idx2token[index]

    def __len__(self):
        return len(self.token2idx)

class TextVectorizer(object):

    def __init__(self, feat_vocab, target_vocab, lang='russian'):
        self.feat_vocab = feat_vocab
        self.target_vocab = target_vocab
        self.lang = lang
        
        
    def vectorize(self, text, vec_type='bag', maxlen=None):

        maxlen = maxlen if maxlen else 512
        if vec_type=='bag':
          res = np.zeros(len(self.feat_vocab), dtype=np.int32)
          for token in word_tokenize(text):
            res[self.feat_vocab.lookup_token(token)] = 1

        elif vec_type=='matrix':
          res = np.zeros((len(self.feat_vocab), maxlen),  dtype=np.int32)
          for ind, it in enumerate(word_tokenize(text)[:maxlen]):
              res[self.feat_vocab.lookup_token(it)][ind] = 1

        elif vec_type=='digits':
          res = np.zeros(maxlen,  dtype=np.int64)
          inds = [self.feat_vocab.lookup_token(it) for it in word_tokenize(text)[:maxlen]]
          res[:len(inds)] = inds
          res[len(inds):] = self.feat_vocab.lookup_token('MASK')

        return res

    @classmethod
    def from_dataframe(cls, df, feat_col, target_col, feat_vocab_size=10000, lang='russian', preprocess=False):

        feat_vocab = Vocabulary(add_unk=True, add_mask=True)
        if target_col:
            target_vocab = Vocabulary(add_unk=False, add_mask=False)
            for it in df[target_col].unique():
                target_vocab.add_token(it)
        else:
            target_vocab = {}

        token_d = Counter()

        
        for text in df[feat_col]:
          # import pdb;pdb.set_trace()
            if preprocess:
              w_l = [it for it in word_tokenize(text.lower(), language=lang) if not it in string.punctuation]
              w_l = [it for it in w_l if not it in stopwords.words(lang)]
            else:
              w_l = word_tokenize(text, language=lang)
                
            token_d.update(w_l)

        for w, count in token_d.most_common(feat_vocab_size):
            feat_vocab.add_token(w)


        return cls(feat_vocab, target_vocab, lang=lang)



def preprocess_text(text, to_lower=True, min_len_token=3, max_len_token=10, stop_words_l=None, stem_lemm_flag=None, save_text=True, save_only_words=True, lang='russian'):

    if save_only_words:
        text = re.sub('(?u)[^\w\s]', '', text)
    text = re.sub(' +', ' ', text)

    if to_lower:
        text = text.lower()

    if stem_lemm_flag=='lemm':
        text = ''.join(Mystem().lemmatize(text))
        stem_lemm_flag=None

    words_l = nltk.tokenize.word_tokenize(text)
    words_l = preprocess_words(words_l, to_lower=False, min_len_token=min_len_token, max_len_token = max_len_token, stop_words_l=stop_words_l, 
                               stem_lemm_flag=stem_lemm_flag, save_only_words=False, lang=lang)

    if save_text:
        return ' '.join(words_l)
    else:
        return words_l

def preprocess_words(words_l, to_lower=True, min_len_token=3, max_len_token=10, stop_words_l=None, 
                     stem_lemm_flag=None, save_only_words=True, lang='russian'):

    if save_only_words:
        punct=list(punctuation)
        words_l = [w for w in words_l if not w.isdigit() and not w in punct]

    if min_len_token:
        words_l = [w for w in words_l if len(w)>=min_len_token]
        
    if max_len_token:
        words_l = [w for w in words_l if len(w)<=max_len_token]

    if stem_lemm_flag=='stem':

        stemmer = SnowballStemmer(lang)
        words_l = [stemmer.stem(w) for w in words_l]

    elif stem_lemm_flag=='lemm':

        lemm = Mystem()
        words_l = [lemm.lemmatize(w) for w in words_l]

    if stop_words_l:
        words_l = [w for w in words_l if not w in stop_words_l]

    if to_lower:
        words_l = [w.lower() for w in words_l]

    return words_l
