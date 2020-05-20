
import os
import json
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from .utils import top_ratio_hit_rate, eval_aucpr,reduce_mem_usage,eval_gini_normalization
import logging 
import warnings
from hyperopt import fmin, tpe, hp, partial,Trials
warnings.filterwarnings('ignore') 

#数据准备

class lgb_optimizer():

    def __init__(self, train_x,train_y,test_x,test_y, default_params, metric='1/1000_hit',logger=None,cat_features=None):
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, stratify=train_y, test_size=0.1,random_state=42)
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y 
        self.test_x= test_x
        self.test_y = test_y
        self.metric=metric
        self.logger = logger
        self.cat_features = cat_features
        self.obj = 0 
        self.best_param={}
        self.best_evals={}
        self.default_params = default_params
        self.default_params.update({"verbose": -1})
        self.space = {"learning_rate":hp.uniform('learning_rate', 5e-3, 2e-1),
                     "max_depth": hp.choice('max_depth',np.arange(3,8).tolist()),  #(min,max,q) = round(uniform(min,max)/q)*q
                     "num_leaves": hp.choice('num_leaves',np.arange(4,60).tolist()), # 
                     "min_data_in_leaf": hp.choice('min_data_in_leaf', np.arange(16,100).tolist()),
                     "bagging_fraction": hp.uniform('bagging_fraction', 0.5,1.0),
                     "feature_fraction": hp.uniform('feature_fraction',0.5,1.0),
                     "min_gain_to_split": hp.uniform('min_gain_to_split', 0,6),
                     "min_data_in_bin": hp.choice('min_data_in_bin',np.arange(10,80,2).tolist()), # 
                     "lambda_l2": hp.uniform('lambda_l2',0.1, 6),
                     "lambda_l1": hp.uniform('lambda_l1',0,5),
                     # "max_cat_threshold": hp.choice('max_cat_threshold',[16,24,32]),
                     }
         
    def lightgbm_factory(self,args):
        params = self.default_params.copy()
        args['num_leaves'] = min(args['num_leaves'], 2**(args['max_depth']-1))
        params.update(args)
        dtrain = lgb.Dataset(self.train_x, label = self.train_y, feature_name='auto',
                        categorical_feature = self.cat_features, weight=None,free_raw_data=False)
        dval = lgb.Dataset(self.val_x, label =self.val_y,feature_name='auto', categorical_feature = self.cat_features, weight=None, 
                        reference=dtrain,free_raw_data=False) 

        evalists =[dtrain,dval]
        evalnames=['train','val']
        eval_result={}

        num_boost_round = params.get('num_boost_round',500)
        early_stopping_round = params.get('early_stopping_round',40)
        model = lgb.train(params, dtrain, num_boost_round, evalists, evalnames, 
                          early_stopping_rounds=early_stopping_round,verbose_eval=40,
                          evals_result=eval_result)
        y_prob = model.predict(self.test_x, num_iteration = model.best_iteration)
        evals={}
        evals['rate1'], _,_, evals['cover'] = top_ratio_hit_rate(self.test_y.values, y_prob, top_ratio=0.001)
        evals['rate5'], _,_,_ = top_ratio_hit_rate(self.test_y.values, y_prob, top_ratio=0.005)
        evals['aucpr'] = eval_aucpr(self.test_y.values, y_prob)
        evals['gini'] = eval_gini_normalization(self.test_y.values, y_prob)
        train_loss = eval_result['train']['binary_logloss'][-1]
        val_loss = eval_result['val']['binary_logloss'][-1]
        metric={'1/1000_hit':evals['rate1'], '5/1000_hit':evals['rate5'],
                'aucpr':evals['aucpr'],'gini':evals['gini'],'30/100_cover':evals['cover']}
        target =metric[self.metric]
  
        if target> self.obj and abs(train_loss - val_loss)<0.02:
            self.obj = target
            params.pop("categorical_column",None)
            self.best_param.update(params)
            self.best_evals.update(evals)
            self.logger.info((' hit_rate = {:.4f}'.format(evals['rate1'])))
            self.logger.info(('hit_rate5 = {:.4f}'.format(evals['rate5'])))
            self.logger.info(('   auc_pr = {:.4f}'.format(evals['aucpr'])))
            self.logger.info(('     gini = {:.4f}'.format(evals['gini'])))
            self.logger.info(self.best_param)
        return -target
    
    def run(self,max_evals=100):
        trials = Trials()
        algo = partial(tpe.suggest, n_startup_jobs=1)
        best = fmin(self.lightgbm_factory, self.space, max_evals=max_evals, algo=tpe.suggest, trials = trials)
        self.logger.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S').center(50,'-'))
        self.logger.info(json.dumps(self.best_param,indent=2))
        print(json.dumps(self.best_param,indent=2))
        print(json.dumps(self.best_evals,indent=2))
        return trials
        # logger.debug(json.dumps(self.best_param,indent=2))
   




# print(trials.trials)
# print(trials.results)
# print(trials.losses() )
# print(trials.statuses() )



