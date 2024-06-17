import nltk

from string import punctuation

from nltk.stem import SnowballStemmer

from pymystem3 import Mystem

import re

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
