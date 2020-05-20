# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:38:13 2019

@author: yuanyuqing163
"""

import pandas as pd
import numpy as np
import os
import json
import operator
import datetime
import pickle
from sklearn.model_selection import train_test_split,StratifiedKFold
#from sklearn.feature_selection import SelectFromModel
import xgboost as xgb

from .category_encoder import TargetEncoder, WoeEncoder
from .utils import (create_feature_map, top_ratio_hit_rate, save_xgb_model)
from .preprocessing import (feature_extract, drop_corr_columns,view_selected_feature)

import warnings
warnings.filterwarnings('ignore')


def dump_config_result(xgb_config, feature_config, best_ntree, feature_nums, 
                       val_hit_rate, enc, type, to_file, info={}, mode='a+'):                       
    log_param = {}
    log_param.update({'feature_config': feature_config})
    log_param.update({'info':info})
    log_param.update({'xgb_config':xgb_config})
    log_param.update({'best_ntree': best_ntree, 'columns':feature_nums,
                      'val_hit_rate':val_hit_rate})
    log_param.update({'encoder': enc})
    log_param.update({'model_type':type})
    log_param.update({'time':datetime.datetime.now().strftime('%Y_%m_%d/%H:%M')})
    json.dump(log_param, open(to_file, mode),indent=2)
    log_param.pop('feature_config')
    print(json.dumps(log_param,indent=2))
    
   
def _xgb_cv_core(X, y, test_x, test_y, params, num_boost_round=400, early_stopping_rounds=40,  n_folds=5,
                shuffle=True, feval=None, maximize=False, imp_file = None, fmap=None, importance_type='gain'):
   
    import time
    if not isinstance(X, pd.DataFrame):
        raise AttributeError('X should be DataFrame')
        
    folds = StratifiedKFold(n_splits=n_folds, shuffle=shuffle)
    
    train_metric={}
    val_metric={}
    hit_rate =[]
    
    eval_metrics = params.get('eval_metric',None)
    test_data = xgb.DMatrix(data=test_x, label=test_y, feature_names=test_x.columns)
    
    importance = pd.DataFrame()  
    features = X.columns.tolist()    
    if imp_file is not None:           
            create_feature_map(features, fmap)
 
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):
        print('===========Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
        train_data = xgb.DMatrix(data=X_train.values, label=y_train.values, feature_names=X.columns)
        valid_data = xgb.DMatrix(data=X_valid.values, label=y_valid.values, feature_names=X.columns)

        watchlist = [(train_data, 'train'), (valid_data, 'val')]
        eval_results ={}
        model = xgb.train(params=params, dtrain=train_data, num_boost_round=num_boost_round,
                          evals=watchlist, 
                          early_stopping_rounds=early_stopping_rounds, 
                          verbose_eval=10, 
                          feval= feval,
                          maximize=maximize,
                          evals_result = eval_results)
        for metric in eval_metrics:
            train_metric.setdefault(metric,[]).append(eval_results['train'][metric][-1])
            val_metric.setdefault(metric,[]).append(eval_results['val'][metric][-1])

        # test top-ratio-hit-rate
        y_prob = model.predict(test_data, ntree_limit=model.best_ntree_limit)
        test_hit_rate, test_top, _,_ = top_ratio_hit_rate(test_y.values, y_prob, top_ratio=0.001)
        print('===val_top_k=%d, val_top_ratio_hit_rate %f\n'%(test_top, test_hit_rate))
        hit_rate.append(test_hit_rate)
        
        if imp_file is not None:           
            fold_imp =model.get_score(fmap=fmap,importance_type=importance_type) 
            # importance_ = sorted(importance_.items(), key=operator.itemgetter(1),reverse=True)
            fold_imp = pd.DataFrame(fold_imp,columns=['feature', 'fscore'])
            # tmp['fscore'] = tmp['fscore']/tmp['fscore'].sum()
            importance = pd.concat([importance,fold_imp],axis=0)
            
    if imp_file is not None:
        importance = importance['fscore'].groupby(importance['feature']).agg({'fscore':'mean'})
        importance['fscore'] = importance['fscore']/importance['fscore'].sum()
        importance=importance.sort_values(by='fscore',ascending=False)
        importance.set_index('feature',inplace=True)
        importance.to_csv(imp_file)
        
    for metric in train_metric:
        print('train-mean-{} = {:.4f}/+-{:.3f}'.format(metric,np.mean(train_metric[metric]), 
                                                np.std(train_metric[metric])))
        print('test-mean-{} = {:.4f}/+-{:.3f}'.format(metric,np.mean(val_metric[metric]),
                                               np.std(val_metric[metric])))
    # print('val-hit-rate={}'.format(hit_rate))
    print('hit-rate=%r'%hit_rate)
    print('val-hit-rate-mean={:.4f}/+-{:.3f}'.format(np.mean(hit_rate),np.std(hit_rate)))    
   
    return  (np.mean(train_metric['auc']), np.mean(val_metric['auc']), 
             '{:.4f}(+-{:.3f})'.format(np.mean(hit_rate),np.std(hit_rate)))
             

def _xgb_core(X, y, test_x,test_y, params, num_boost_round=400, early_stopping_rounds=40,
                feval=None, maximize=False, model_path=None, fmap=None, imp_file=None,
                imp_type=None, tmp_path=None):
    
    print('xgb_model'.center(40,'-'))         
    train_x, val_x, train_y, val_y = train_test_split(X, y, stratify=y,  test_size=0.1,random_state=42)
    dtrain = xgb.DMatrix(train_x.values,label = train_y.values, weight=None)
    dval = xgb.DMatrix(val_x.values, label =val_y.values, weight=None)
    dtest = xgb.DMatrix(test_x.values, label =test_y.values, weight=None)
    
    #------------------------config-----------------------#  
    evalists =[(dtrain,'train'),(dval,'val')] # the last one will used for early stopping 
    eval_result={}
    model = xgb.train(params, dtrain, num_boost_round, evalists, feval = feval, maximize= maximize,
                      early_stopping_rounds=early_stopping_rounds, verbose_eval=10,
                      evals_result = eval_result)
                      
    train_auc = eval_result['train']['auc'][-1]
    val_auc = eval_result['val']['auc'][-1]
    
    features = train_x.columns.tolist() 
    save_xgb_model(model, model_path, features=features, fmap=fmap, 
                  imp_path=imp_file, importance_type = imp_type)
   
    #--------------------prediction and valiation -----------------
    test_y_prob = model.predict(dtest, ntree_limit = model.best_ntree_limit)
    test_hit_rate, test_top, _,_ = top_ratio_hit_rate(test_y.values,test_y_prob, top_ratio=0.001)
    print('===test_top_k=%d, test_top_ratio_hit_rate %f\n'%(test_top, test_hit_rate))
    
    # calc_threshold_vs_depth(test_y.values,test_y_prob, stats_file=None)
  
    return train_auc, val_auc, test_hit_rate, model.best_ntree_limit
    
    
def xgb_train(train_x,train_y,test_x,test_y,feature_config, xgb_config, input_file, output_file, cv_folds=0,
                   nopre=False, save_clean_data=False):
    #-------------------------read_data-------------------------
    if  not nopre: 
        anotation_dict = {}
        feature_extract(train_x,feature_config, input_file, output_file, anotation_dict, 
                            unuse_cols=[],drop_less=feature_config['drop_less'],imp_inplace=True)

        enc = WoeEncoder()
        enc.fit(train_x, train_y, extra_numeric_cols=None)
        train_x = enc.transform(train_x)
        #------------------------------drop corrlation columns -------------------
        drop_corr_cols = drop_corr_columns(train_x, corr_file = output_file['corr_path'],
                                           corr_threshold = feature_config['corr_threshold'])
        print('drop_corr_cols:',len(drop_corr_cols), drop_corr_cols)
        enc.encode_cols = [col for col in enc.encode_cols if col not in drop_corr_cols]
        print('\nencode_cols=%s'%enc.encode_cols)

        model_columns = train_x.columns.tolist()
        json.dump(enc.mapping, open(output_file['encoder_path']+'_map','w+'), indent=2)
        pickle.dump(enc, open(output_file['encoder_path']+'_pkl','wb+'))
        
        view_selected_feature(train_x, out_file= output_file['select_feature_path'],
                      encoder_cols= enc.encode_cols, anotation_dict=anotation_dict) 
        #------------------------------validation_data----------------------------
        test_x = test_x[model_columns]
        test_x = enc.transform(test_x)
        print('test-data:', test_x.shape)
        #-----------------------save for cv---------------------------------------
        if save_clean_data: 
            pd.concat([train_x, train_y],axis=1).to_pickle(input_file['cv_train_file'])
            pd.concat([test_x,test_y],axis=1).to_pickle(input_file['cv_val_file'])
    else:
        train_x = pd.read_pickle(input_file['cv_train_file'])
        train_y = train_x.iloc[:,-1]

        train_x = train_x.iloc[:,:-1]
        print(train_x.shape)
        
        test_x= pd.read_pickle(input_file['cv_val_file'])
        test_y = test_x.iloc[:,-1]
        test_x = test_x.iloc[:,:-1]
        print(test_x.shape)
        
    #----------------------xgboost model ---------------------
    print('model-training'.center(60,'-'))
    params = xgb_config.copy()
    num_round = params.pop('num_round')
    early_stopping_rounds= params.pop('early_stopping_rounds')
    feval = params.pop('feval')
    maximize = params.pop('maximize')
    imp_file = input_file['imp_file'] if feature_config['xgb_importance_revise'] else None
    if cv_folds:
        train_metric, test_metric, val_hit_rate = _xgb_cv_core(
                                              train_x, train_y, test_x, test_y, params, 
                                              num_boost_round = num_round,
                                              early_stopping_rounds = early_stopping_rounds,  
                                              n_folds = cv_folds,
                                              feval = feval,
                                              maximize = maximize,
                                              imp_file = imp_file,
                                              fmap = input_file['fmap'],
                                              importance_type =feature_config['xgb_importance_type'])

    else:
        train_auc, test_auc, val_hit_rate, best_ntree = _xgb_core(
                                              train_x, train_y, test_x,test_y, params, 
                                              num_boost_round=num_round,
                                              early_stopping_rounds=early_stopping_rounds,
                                              feval=feval, maximize=maximize,
                                              model_path=output_file['model_path'],
                                              fmap=input_file['fmap'],
                                              imp_file=imp_file,
                                              imp_type=feature_config['xgb_importance_type'],
                                              tmp_path=output_file['tmp_path']) 

    xgb_config.pop('feval') # pop feval as fun object can not dump with json 
    xgb_config.pop('maximize')
    dump_config_result(xgb_config, feature_config,
                       best_ntree = 'null' if cv_folds else best_ntree, 
                       feature_nums = train_x.columns.size, 
                       val_hit_rate=val_hit_rate,  enc = 'WoeEncoder', 
                       type = 'xgb',
                       to_file = output_file['log_file'],
                       info={})

if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', type=int, help='cross validation folds') 
    parser.add_argument('--nopre', action='store_true', help='no data preprocessing') # default false
    parser.add_argument('--save', action ='store_true') #default false
    args = parser.parse_args(sys.argv[1:])
    cv_folds= args.cv if args.cv else 0
    nopre = args.nopre
    save_clean_data = (args.save) 
    if nopre:
        print('use clean data without feature extract'.center(50,'-'))
    else:
        print('use rawinput with feature extract'.center(50,'-'))
        if save_clean_data:
            print('save clean data'.center(50,'-'))

    feature_config, xgb_config, input_file, output_file = load_xgb_config()
    # feature_config, xgb_config, input_file, output_file = load_xgb_bj_ge18_config()
    xgb_train(feature_config, xgb_config, input_file, output_file,
                    cv_folds = cv_folds,
                    nopre = nopre,
                    save_clean_data = save_clean_data)
   





