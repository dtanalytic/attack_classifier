import operator
import math
import random
from deap import base
from deap import creator
from deap import gp

from deap import tools
from sklearn.metrics import log_loss

from sklearn.model_selection import cross_val_score

def get_gen_proba(f, X, filt_cols):
    y_pred = [f(*point) for point in X[filt_cols].values.tolist()]
    y_pred = [min(0.99, it) for it in y_pred]
    y_pred = [max(0, it) for it in y_pred]
    return y_pred



def get_toolbox(N):
    
    pset = gp.PrimitiveSet("MAIN", arity=N)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addEphemeralConstant('a',lambda: random.uniform(0,1))
    pset.addEphemeralConstant('b',lambda: random.uniform(0,1))

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
                  pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=1)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                    toolbox.expr)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)



    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=0)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))
    
    return toolbox
    

def fake_splitter(df):
    yield (df[(df.split=='tr')].index, df[(df.split=='val')].index)

class BFSThres():
    
    def __init__(self, estimator, scoring, cv = 5, thresh=0.005, metric_sign=1):
        self.estimator = estimator
        self.thresh = thresh
        self.cv = list(cv) if not isinstance(cv, int) else cv
        self.scoring = scoring
        self.metric_sign = metric_sign
        
    def update_estimator(self, X):
        return self.estimator
    
    def fit(self, X, y, verbose=False):
        feat_num = len(X.columns)
        self.mask = [True]*feat_num
        self.sc_pr = cross_val_score(self.estimator, X, y, cv=self.cv, scoring = self.scoring).mean()
        first_qual = self.sc_pr
        if verbose:
            print(f"Начальное качество на признаках: {X.columns[self.mask]} - {first_qual}")
            
        for i in range(feat_num-1, -1, -1):
            self.mask[i]=False
            if verbose:
                print(f"На рассмотрении признак {X.columns[i]}:")
                
            if len(X.columns[self.mask])==0:
                if self.estimator._estimator_type=='classifier':
                    from sklearn.dummy import DummyClassifier
                    dummy_model = DummyClassifier(strategy='most_frequent')
                elif self.estimator._estimator_type=='regressor':
                    from sklearn.dummy import DummyRegressor
                    dummy_model = DummyRegressor(strategy='mean')

                sc_cur = cross_val_score(dummy_model, X[X.columns[self.mask]], y, cv=self.cv, scoring = self.scoring).mean()
            else:
                sc_cur = cross_val_score(self.update_estimator(X), X[X.columns[self.mask]], y, cv=self.cv, scoring = self.scoring).mean()
            
            qual_ratio = (sc_cur-self.sc_pr)/self.sc_pr*self.metric_sign

            if verbose:
                print(f"Текущее качество - {sc_cur}, прошлое: {self.sc_pr}, улучшение: {qual_ratio:.2%}")
            if qual_ratio>=-self.thresh:
                self.sc_pr = sc_cur
                if verbose:
                    print(f"Новый набор признаков: {X.columns[self.mask]}")
                # self.estimator = self.update_estimator(X)
            else:
                self.mask[i]=True
                
            if verbose:
                print(f"Отсеянные признаки f{X.columns[[not it for it in self.mask]]} \n")


        if verbose:
                print(f"Начальное качество было - {first_qual}, итоговое на признаках: {X.columns[self.mask]} - {self.sc_pr}")