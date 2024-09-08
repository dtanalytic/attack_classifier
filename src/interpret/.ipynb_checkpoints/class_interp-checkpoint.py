from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd

# data must contain labels col and col of sents
def class_word_stat(data, col, word):

    return data.explode('labels').assign(has_sent = lambda x: x[col].str.contains(word)).pivot_table(index='labels', values='has_sent', aggfunc=['count', 'sum', 'mean']).sort_values(by=('mean', 'has_sent'), ascending=False)

class BayesExplain():

    def __init__(self, data, feat_data, mlb):
        
        self.data = data
        self.feat_data = feat_data.drop(columns='attack_pred')
        self.mlb = mlb
        self.col_l = self.feat_data.columns
        
    
    def fit(self):

        tr_idx = self.data.query('split=="tr"').index
              
        model = OneVsRestClassifier(MultinomialNB(alpha=0.0001, fit_prior=False))
        
        model.fit(self.feat_data.loc[tr_idx].astype(np.float16).values, np.array(self.data.loc[tr_idx, 'target'].values.tolist()))
        self.model = model
        
        # формируем массив слов примечательных для атак
        feat_cols_ar = np.eye(len(self.col_l))

        feat_cols_probs = model.predict_proba(feat_cols_ar)
        self.col_prob_df = pd.DataFrame(feat_cols_probs, index=self.col_l, columns=self.mlb.classes_)

        self.cls_d = {}
        for num, cls in enumerate(self.mlb.classes_):
            idx = np.argsort(feat_cols_probs[:, num])
            self.cls_d[cls] = [(self.col_l[it], feat_cols_probs[:, num][it]) for it in idx]
        
        # байесовские вероятности для каждого слова в классе
        feat_bayes_probs_df = pd.DataFrame(index=self.feat_data.columns)
        for i, cls in enumerate(self.mlb.classes_):
            probs = self.model.estimators_[i].feature_log_prob_[1]
            probs = np.exp(probs)
            feat_bayes_probs_df[cls] = probs 

        self.feat_bayes_probs_df = feat_bayes_probs_df
        
        return self
    
    def cls_explain(self, name, k=10):
        return self.cls_d['impact'][-k:]

    def word_explain(self, word):
        return self.col_prob_df.loc[word]

    def get_predictions(self):
        return self.model.predict_proba(self.feat_data.astype(np.float16).values)
        
    def text_explain(self, idx, exp_type='algo'):
        '''
      Если рассматривать как итоговый прогноз произведение вероятностей прогнозирования слов для класса, то больше подходит exp_type=='algo'. Так как 
      exp_type=='word_preds' выводит вероятности слов относительно нулевого класса (то есть вероятности приводятся на сумму, что сильно влияет на пропорцию разности значений и, соответственно, на итоговое произведение), а в реализаци алгоритма сначала считается произведение вероятностей слов для каждого класса (возведенное в степень количества и умноженное на априоры классов), которое затем делится на сумму.
        '''
        word_l = self.feat_data.iloc[idx,:].loc[lambda x: x>0].index.tolist()
        
        if exp_type=='word_preds':
            col_prob_df = self.col_prob_df
        if exp_type=='algo':
            col_prob_df = self.feat_bayes_probs_df
        
        res_df = col_prob_df.loc[word_l[0]].to_frame()
        for i in range(1, len(word_l)):
            res_df = res_df.join(col_prob_df.loc[word_l[i]].to_frame())
        res_df['prod'] = res_df.prod(axis=1)
        return res_df.sort_values(by='prod', ascending=False)


