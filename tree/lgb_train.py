#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 20:38:13 2019

@author: yuanyuqing163
"""

import pandas as pd
import numpy as np
import os
import json
import datetime
import pickle
import shutil
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.utils import shuffle
import lightgbm as lgb
from .category_encoder import LabelEncoder
# from datasrc import get_anotation_dict,view_selected_feature
from .utils import (create_feature_map, top_ratio_hit_rate, save_lgb_model,calc_threshold_vs_depth)
from .preprocessing import (feature_extract,drop_corr_columns,drop_model_importance_columns,
                             view_selected_feature, calc_permutation_importance) 
                            # drop_category_columns, drop_low_iv_columns,
                           #drop_null_columns,drop_model_importance_columns,

import warnings
warnings.filterwarnings('ignore')  
#-------------------------read_data------------------------------------

def dump_config_result(lgb_config, feature_config, model_iteration, feature_nums, 
                        train_auc, test_auc, val_hit_rate, enc, type, to_file, info={}, mode='a+'):                       
    log_param = {}
    log_param.update({'feature_config': feature_config})
    log_param.update({'info':info})
    log_param.update({'lgb_config':lgb_config})
    log_param.update({'best_iteration': model_iteration,
                      'columns':feature_nums,
                      'train_auc':train_auc,
                      'test_auc':test_auc,
                      'val_hit_rate':val_hit_rate})
    log_param.update({'encoder': enc})
    log_param.update({'model_type':type})
    log_param.update({'time':datetime.datetime.now().strftime('%Y_%m_%d/%H:%M')})
    json.dump(log_param, open(to_file, mode),indent=2)
    log_param.pop('feature_config')
    print(json.dumps(log_param,indent=2))
    
  
def _lgb_cv_core(X, y, test_x, test_y, params, num_boost_round=400, early_stopping_rounds=40,  n_folds=5,
                shuffle=True, cat_features=None,feval=None):
    import time
    
    if not isinstance(X, pd.DataFrame):
        raise AttributeError('X should be DataFrame')
    print('lgb_cv_model'.center(40,'-'))
    folds = StratifiedKFold(n_splits=n_folds, shuffle=shuffle)
    
    train_metric={}
    val_metric={}
    hit_rate =[]
    
    eval_metrics = params.get('metric',None)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)): 
        print('Fold', fold_n, 'started at', time.strftime('%H-%M'))
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
        train_data = lgb.Dataset(data=X_train, label=y_train, feature_name='auto',
                                 categorical_feature = cat_features)
        valid_data = lgb.Dataset(data=X_valid, label=y_valid, feature_name='auto',
                                 categorical_feature=cat_features, reference=train_data)
                                 
        evalists =[train_data, valid_data]
        evalnames=['train','val']
        eval_results={}
        model = lgb.train(params, train_data, num_boost_round, evalists, evalnames, 
                        early_stopping_rounds=early_stopping_rounds, evals_result=eval_results,
                        verbose_eval = 10,feval=feval)
                        
        print('===best_iteration=%d'%model.best_iteration)               
        for metric in eval_metrics:
            train_metric.setdefault(metric,[]).append(eval_results['train'][metric][-1])
            val_metric.setdefault(metric,[]).append(eval_results['val'][metric][-1])

        # test top-ratio-hit-rate
        y_prob = model.predict(test_x, num_iteration = model.best_iteration)
        val_hit_rate, val_top, _,_ = top_ratio_hit_rate(test_y.values, y_prob, top_ratio=0.001)
        print('===val_top_k=%d, val_top_ratio_hit_rate %f\n'%(val_top, val_hit_rate))
        hit_rate.append(val_hit_rate)
    
    for metric in train_metric:    
        print('train-mean-{} = {:.4f}(+-{:.3f})'.format(metric,np.mean(train_metric[metric]),np.std(train_metric[metric])))   
        print('test-mean-{} = {:.4f}(+-{:.3f})'.format(metric,np.mean(val_metric[metric]),np.std(val_metric[metric])))  
    print('val-hit-rate-mean={:.4f}(+-{:.3f})'.format(np.mean(hit_rate),np.std(hit_rate)))
    print('hit-rate=%r'%hit_rate)                  
    return  np.mean(train_metric['auc']), np.mean(val_metric['auc']), '{:.4f}(+-{:.3f})'.format(np.mean(hit_rate),np.std(hit_rate))
    
def _lgb_core(X, y, test_x,test_y,params, num_boost_round=400, early_stopping_rounds=40, fobj=None,feval=None,        
              learning_rates=None, cat_features = None, model_path=None, imp_type='gain', imp_path=None):
    print('lgb_model'.center(40,'-'))         
    train_x, val_x, train_y, val_y = train_test_split(X, y, stratify=y,  test_size=0.1,random_state=42)

    dtrain = lgb.Dataset(train_x, label = train_y, feature_name='auto',
                        categorical_feature = cat_features, weight=None)
    dval = lgb.Dataset(val_x, label =val_y,feature_name='auto',
                        categorical_feature = cat_features, weight=None, reference=dtrain)      
    # dtest = lgb.Dataset(test_x, label =test_y,feature_name='auto',
                        # categorical_feature = cat_features, weight=None, reference=dtrain)   
                         
    evalists =[dtrain,dval]
    evalnames=['train','val']
    eval_result ={}    
    model = lgb.train(params, dtrain, num_boost_round, evalists, evalnames, 
                      early_stopping_rounds=early_stopping_rounds,
                      verbose_eval = 20, evals_result = eval_result,
                      feval=feval,fobj=fobj,
                      learning_rates=learning_rates)                                         
    print('best_iteration=%d'%model.best_iteration)
    
    features = train_x.columns.tolist()
    save_lgb_model(model, model_path, features=features, imp_path=imp_path, imp_type=imp_type,
                    cat_features=cat_features)
    
    #-----------------------------prediction and valiation -----------------
    tree_info = model.dump_model()
    json.dump(tree_info, open(os.path.join(os.path.split(model_path)[0],'lgb.model_raw'),'w+'),indent=2) 

    # y_prob = model.predict(span_val_x, num_iteration = model.best_iteration)
    # val_hit_rate, val_top,index= top_ratio_hit_rate(span_val_y.values, y_prob, top_ratio=0.001)
    # print('val_top_k=%d, val_top_ratio_hit_rate %f'%(val_top, val_hit_rate))

    test_y_prob = model.predict(test_x, num_iteration = model.best_iteration)
    test_hit_rate, test_top, index, _= top_ratio_hit_rate(test_y.values, test_y_prob, top_ratio=0.001)
    print('test_top_k=%d, test_top_ratio_hit_rate %f\n'%(test_top, test_hit_rate))
    
    _,_,cover,_ = calc_threshold_vs_depth(test_y.values, test_y_prob)
    
    # permutation_importance(model, test_x, test_y, eval_result['val']['aucpr'][-1], folds=2, file='../tmp/lgb_permutation_importance.csv')
    
    return eval_result['train']['auc'][-1], eval_result['val']['auc'][-1], test_hit_rate,  model.best_iteration,index, cover
    

    
def lgb_train(train_x,train_y,val_x, val_y, feature_config, lgb_config, input_file, output_file, category_features,cv_folds=0, nopre=False,  save_clean_data=False,use_local_data=False):
    # logger = getLogger(log_to_file=output_file['log_file'])
    #-------------------------read_data--------------------------------  
    info ={}
    if not nopre:

        anotation_dict={}
        #-----------------------config feature here--------------------------
        feature_extract(train_x, feature_config,  input_file, output_file, 
                                       anotation_dict, unuse_cols=[], drop_less=feature_config['drop_less'],
                                       imp_inplace=True)
        print('extracted:', train_x.shape)

        #--------------------------drop correlation--------------------------
        drop_corr_cols = drop_corr_columns(train_x, corr_file = output_file['corr_path'],
                                           corr_threshold = feature_config['corr_threshold'])
        print('drop corr cols [%d] = %r'%(len(drop_corr_cols), drop_corr_cols))

        
        #--------------------------int encoder columns ----------------------
        # df = pd.read_csv(input_file['int_encoder_path'])
        # int_encode_cols = df['feature'].drop_duplicates().values.tolist()    
        # int_encode_cols = [col for col in int_encode_cols if col in data.columns]  
        # print('int type need category encoder = %s'% int_encode_cols)
        
        model_columns = train_x.columns.tolist()
            
        # -----------------------------label encoder------------------------
        enc = LabelEncoder()
        enc.fit(train_x, y=None, extra_numeric_cols=None)
        
        # encoder_path = "/home/yuanyuqing163/hb_hnb_0310/model/lgb_encoder_dl_pkl_imp109_1224_0.91"
        # with open(encoder_path,'rb') as fp:
        #     enc = pickle.load(fp)
        train_x = enc.transform(train_x)
        category_features = enc.encode_cols
        print('---train_shape---', train_x.shape)
        print('category_feature_len=%d'%len(category_features))
        view_selected_feature(train_x, out_file= output_file['select_feature_path'],
                              encoder_cols= enc.encode_cols, anotation_dict=anotation_dict)
        json.dump(enc.mapping, open(output_file['encoder_path']+'_map','w+'), indent=2)
        pickle.dump(enc, open(output_file['encoder_path']+'_pkl','wb+'))

        #------------------------------validation_data----------------------------
        val_x = val_x[model_columns]
        val_x = enc.transform(val_x)
        print('---val-shape---', val_x.shape)

        # -----------------------------test_date ----------------------------------
        # test_x = test_x[model_columns]
        # test_x = enc.transform(test_x)
        # print('---test-shape---',test_x.shape)
        #--------------------fill NA of double cols with zero---------------------
        #-----------------------save for cv---------------------------------------
        if save_clean_data:
            pd.concat([train_x, y],axis=1).to_pickle(input_file['cv_train_file'])
            pd.concat([val_x,val_y],axis=1).to_pickle(input_file['cv_val_file'])
            np.savetxt(input_file['category_feature_path'], category_features, fmt='%s',delimiter='\t')

    if use_local_data:
        train_x = pd.read_pickle(input_file['cv_train_file'])
        train_y = train_x.iloc[:,-1]
        train_x = train_x.iloc[:,:-1]
        print('---train_shape---', train_x.shape)

        val_x= pd.read_pickle(input_file['cv_val_file'])
        val_y = val_x.iloc[:,-1]
        val_x = val_x.iloc[:,:-1]
        print('---val_shape---', val_x.shape)
        category_features = np.loadtxt(open(input_file['category_feature_path'],'r'), 
                                        dtype=str,  delimiter='\t')
        category_features = category_features.tolist()
        # dtype_to_encode=['object','category']
        # category_features = train_x.select_dtypes(include=dtype_to_encode).columns.tolist()
        print('=category_feature_len=%d'%len(category_features)) 
    #------------------------config-----------------------#
    params = lgb_config.copy()
    num_round = params.pop('num_boost_round')
    print(json.dumps(params,indent=2))
    early_stopping_rounds= params.pop('early_stopping_round')
    fobj = params.pop('fobj',None)
    feval = params.pop('feval',None)
    learning_rates = params.pop('learning_rates')
    if cv_folds:
        train_auc, test_auc, val_hit_rate = _lgb_cv_core(
                                              train_x, train_y, val_x, val_y, params, 
                                              num_boost_round=num_round,
                                              early_stopping_rounds=early_stopping_rounds,  
                                              n_folds=cv_folds,
                                              cat_features=category_features)                                                                          
    else:
        imp_path = input_file['lgb_imp_file'] if feature_config['lgb_importance_revise'] else input_file['lgb_imp_tmp_file']
        train_auc, test_auc, val_hit_rate, iteration, index, cover = _lgb_core(
                                             train_x, train_y, val_x, val_y, params, 
                                             num_boost_round=num_round, 
                                             early_stopping_rounds=early_stopping_rounds,                                                               
                                             cat_features = category_features, 
                                             model_path=output_file['model_path'], 
                                             imp_type=feature_config['lgb_importance_type'], 
                                             imp_path= imp_path,
                                             feval=feval,
                                             fobj=fobj, learning_rates=learning_rates)
        # X_err = val_x_ori.iloc[index,:].reset_index(drop=True)
        # y_err = val_y.iloc[index].reset_index(drop=True)
        # pd.concat([X_err, y_err],axis=1).to_csv(os.path.split(output_file['null_path'])[0]+'/top_predict_samples.csv')
                                             
    lgb_config.pop('feval')
    lgb_config.pop('fobj')
    lgb_config.pop('learning_rates')
    dump_config_result(lgb_config, feature_config, 
                       model_iteration = 'null' if cv_folds else iteration, 
                       feature_nums = train_x.columns.size, 
                       train_auc = train_auc,
                       test_auc = test_auc, 
                       val_hit_rate = val_hit_rate,
                       enc = 'LabelEncoder', 
                       type = 'lgb',
                       to_file = output_file['log_file'],
                       info={})

    src_model_path = os.path.split(output_file['model_path'])[0]
    base_path, cur_model_folder = os.path.split(src_model_path)
    if not nopre:
        dst_model_path = os.path.join(base_path,'backup',cur_model_folder+datetime.datetime.now().strftime('_%m%d')+
                        '_'+str(train_x.columns.size)+'_'+'{:.4f}'.format(val_hit_rate)+'_local')
    else:
        dst_model_path = os.path.join(base_path,'backup',cur_model_folder+datetime.datetime.now().strftime('_%m%d')+
                        '_'+str(train_x.columns.size)+'_'+'{:.4f}'.format(val_hit_rate))
    if os.path.exists(dst_model_path):
        shutil.rmtree(dst_model_path)
    shutil.copytree(src_model_path,dst_model_path)
    
    with open(os.path.join(dst_model_path,'eval.metric'),'a+') as fp:
        fp.write(str(val_hit_rate)+'\t'+str(cover)+'\n')

    return dst_model_path
   
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
    
    feature_config, xgb_config, input_file, output_file = load_lgb_config()
    # feature_config, xgb_config, input_file, output_file = load_lgb_bj_ge18_config()

    lgb_train(feature_config, xgb_config, input_file, output_file, cv_folds=cv_folds, 
                   nopre= nopre,
                   save_clean_data=save_clean_data)
    
    # python lgb_train.py --nopre  --cv 5
    # python lgb_train.py --save
   

