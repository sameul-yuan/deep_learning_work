# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:02:32 2019

@author: yuanyuqing163
"""
import os
import numpy as np
import pandas as pd
# from .utils import safe_cast_float2int_by_fillna
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
def drop_null_columns(df, null_percent=0.8, inplace=True, 
                      null_columns_store=None,
                      anotation_dict={}):
    
    null_cnt = df.isnull().sum(axis=0)/df.index.size
    null_cnt = null_cnt[null_cnt>=null_percent]
    null_cols = null_cnt.index.tolist()
    if inplace:
        df.drop(null_cols,axis=1, inplace =True)

    if null_columns_store is not None:
        tmp = pd.Series()
        for col in null_cols:
            tmp = tmp.append(pd.Series(anotation_dict.get(col,'N'),index=[col]))
        pd.concat([null_cnt, tmp], axis=1).rename(columns={0:'null_ratio',1:'anotate'}).\
                 sort_values(by='null_ratio',ascending=False).\
                 to_csv(null_columns_store, encoding='gbk')   
    print('-------------------\ndrop null columns = %d'%len(null_cols))
    return null_cols

    
def drop_category_columns(df, category_threshold = 50, inplace=True,
                          category_columns_store=None,
                          anotation_dict={}, int_encoder= []):
    
    #------------drop columns with too much categories and only one category
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols += [col for col in int_encoder if col in df.columns]       
    cat_cnts = df[cat_cols].apply(lambda x: x.value_counts(dropna=False).size)
    drop_cols = cat_cnts[(cat_cnts>category_threshold) | (cat_cnts==1)].index.tolist()
    # numeric feature
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cnts = df[numeric_cols].apply(lambda x: x.value_counts(dropna=False).size) #include np.nan
    drop_cols += numeric_cnts[numeric_cnts==1].index.tolist()
    
    # remove duplicated
    drop_cols = sorted(list(set(drop_cols)), key=drop_cols.index)    
                
    if category_columns_store is not None:
        tmp_val=[] 
        tmp_dtype=[]
        tmp_cat=[]
        tmp_meaning=[]
        for col in drop_cols:
            tmp_val.append(df[col].sample(10,replace=False).values.tolist())
            tmp_dtype.append(df[col].dtype.name)
            tmp_cat.append(df[col].unique().size)
            tmp_meaning.append(anotation_dict.get(col,'N'))
        pd.DataFrame({'anotate':tmp_meaning,'dtype': tmp_dtype, 'category':tmp_cat,
                        'samples':tmp_val},index=drop_cols).\
                     sort_values(by='category',ascending=False).\
                     to_csv(category_columns_store, encoding='gbk')
    if inplace:         
        df.drop(columns=drop_cols, inplace=True)
        
    print('-------------------\ndrop category columns= %d'%len(drop_cols))       
    return drop_cols

def drop_low_iv_columns(df, iv_file, iv_threshold=0.02, inplace=True,
                        low_iv_store=None,
                        anotation_dict={}):
    
    iv = pd.read_table(iv_file, sep=',', header=None, names=['feature','IV'])
    iv.set_index('feature',inplace=True)
    drop_iv = iv[iv['IV']<=iv_threshold]
    columns = df.columns
    # first drop columns not inlucded in iv table
    # drop_not = df.columns[~df.columns.isin(iv.index)].tolist()
    drop_not =[]
    print('-------------------\ncolumns not in iv={}'.format(len(drop_not)))
    # drop column included in df
    drop_iv = drop_iv[drop_iv.index.isin(columns)]
    drop_columns = drop_not + drop_iv.index.tolist()
    
    if inplace:
        df.drop(drop_columns,axis=1, inplace=True)
        
    if low_iv_store is not None:
        anotation = pd.Series()
        ivv = pd.Series()
        for col in drop_columns:
            anotation = anotation.append(pd.Series(anotation_dict.get(col,'N'),index=[col]))
            #ivv = ivv.append(pd.Series(iv[col] if col in iv else 'N', index=[col]))
        pd.concat([drop_iv, anotation], axis=1).rename(columns={1:'anotate'}).\
                 to_csv(low_iv_store, encoding='gbk')   
    print('drop iv columns = %d'%len(drop_columns))
    
    return drop_columns
        
def drop_model_importance_columns(df, imp_file, score_quantile=None,
                                  score_threshold = None, drop_less =False,
                                  inplace=True, **read_config):
    _default_config =dict(sep=',')
    _default_config.update(**read_config)
    imp = pd.read_table(imp_file, **_default_config)
    imp.columns=['feature','score']
    if score_quantile is None and score_threshold is None:
        raise ValueError('either importance threshold or quantile need set')
    if score_quantile:
        assert 0.0<=score_quantile<=1.0, 'score_quantile should be in [0,1]'
        score_threshold = imp['score'].quantile(score_quantile)
    print('-------------------\nmodel_importance_threshold=%f'%score_threshold)
    
    if drop_less:
        imp = imp[imp['score']<score_threshold]
        drop_cols = imp[imp['feature'].isin(df.columns)]['feature'].tolist()
    else:
        features = imp[imp['score']>=score_threshold]['feature']
        drop_cols = df.columns.difference(features)

     
    if inplace:
        print('drop_model_importance_columns=%d'%len(drop_cols))
        df.drop(columns=drop_cols, inplace=True)
    
    return drop_cols
        
def drop_corr_columns(df, corr_file, corr_threshold=0.99, inplace=True):
    import operator
    corr = df.corr()
    upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))
    # pos_cols  = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    # neg_cols = [column for column in upper.columns if any(upper[column] < -corr_threshold)]
    # drop_cols = list(set(pos_cols + neg_cols))
    # df.drop(columns = drop_cols, inplace=True)
    #save info
    mask_mat = (upper > corr_threshold) | (upper < -corr_threshold)
    col1 = np.where(mask_mat)[0]
    col2 = np.where(mask_mat)[1]  

    drop_cols=[]
    keep_cols=[]    
    if len(col1)>0:
        pairs = zip(col1, col2)    
        pairs = sorted(pairs, key = operator.itemgetter(0))
        col1, col2 = zip(*pairs)
        col1_name = upper.index[list(col1)]
        col2_name = upper.index[list(col2)]
    
        corr_value = [ ]
        for col1_, col2_ in zip(col1_name, col2_name):
            if col1_ not in drop_cols:
                drop_cols.append(col2_)
                keep_cols.append(col1_)
                corr_value.append(upper.loc[col1_, col2_])
                
        pd.DataFrame({'col1':keep_cols,'col2':drop_cols,'corr':corr_value}).sort_values(
                    by='col1').to_csv(corr_file, index=False)  
                    
        drop_cols =list(set(drop_cols))
        if inplace:
            df.drop(columns= drop_cols, inplace=True)
           
    return drop_cols
    
               
def feature_extract(data,feature_config:dict, input_file:dict, output_file:dict,
                        anotation_dict:dict, unuse_cols:list,drop_less=False, imp_inplace=True): 
    """
    featuer extraction base on null value, category_threshold, and iv-value, 
        and xgb_feature_importance 
    """      
    # drop null 
    drop_null_columns(data, null_percent=feature_config['null_percent'], 
                    null_columns_store = output_file['null_path'], 
                    anotation_dict = anotation_dict,inplace=True)
                    
    # recover float to int 
    # df = pd.read_csv(input_file['int_encoder_path'])
    # int_encode_cols = df['feature'].drop_duplicates().values.tolist()     
    # int_encode_cols = safe_cast_float2int_by_fillna(data, cols= int_encode_cols)     
    drop_category_columns(data,category_threshold=feature_config['category_threshold'],
                    category_columns_store = output_file['category_path'],
                    anotation_dict = anotation_dict, inplace=True, 
                    int_encoder =[])
                    
    ## drop low information value attrs
    if input_file['iv_file']:
        drop_low_iv_columns(data,iv_file = input_file['iv_file'], iv_threshold =feature_config['iv_threshold'],
                        low_iv_store = output_file['low_iv_path'], anotation_dict = anotation_dict, 
                        inplace=True)
     
    # drop low importance attrs based on xgboost     
    if not feature_config['xgb_importance_revise']:
        drop_model_importance_columns(data,imp_file=input_file['imp_file'],
                                          score_threshold=feature_config['imp_threshold'],
                                          drop_less = drop_less, inplace=imp_inplace)
    # drop unusable attrs                                
    drop_cols = [col  for col in unuse_cols if col in data.columns]
    data.drop(drop_cols, axis=1, inplace=True) # can drop [] columns
                                              
    print('after extraion: columns=%d'%data.columns.size)
    
    # return [col for col in int_encode_cols if col in data.columns] 
    
def view_selected_feature(df, out_file, encoder_cols=None, anotation_dict={}):   
    """
    df: the df befor category encoder
    """      
    sel_cols = df.columns.tolist()
    tmp_val =[]
    tmp_dtype=[]
    tmp_cat=[]
    tmp_meaning=[]
    tmp_encoder=[]
    encoder_cols = [] if encoder_cols is None else encoder_cols
    for col in sel_cols:
        type_ = df[col].dtype.name
        tmp_val.append(df[col].sample(20,replace=False).values.tolist())
        tmp_dtype.append(type_)
        tmp_cat.append(df[col].unique().size if ((type_.startswith('object')) |
                            (type_.startswith('int')) | 
                            (type_.startswith('category'))) else np.nan)
        tmp_meaning.append(anotation_dict.get(col,'N'))
        tmp_encoder.append('encoder' if col in encoder_cols else np.nan)
        
    pd.DataFrame({'meaning':tmp_meaning,'dtype': tmp_dtype, 'category':tmp_cat,
                  'encoder':tmp_encoder, 'samples':tmp_val},index=sel_cols).\
                 sort_values(by='category').\
                 to_csv(out_file, encoding='gbk')
    np.savetxt(os.path.splitext(out_file)[0]+'.txt', sel_cols, fmt='%s', delimiter='\t')
    print('total select field = %d'%df.columns.size)
    
def get_anotation_dict(anotation_file):
    import json
    with open(anotation_file, 'r') as fp:
        anotation_dict = json.load(fp)
        return anotation_dict

def calc_permutation_importance(model, data,labels, baseline, folds=3,file=None):
    from sklearn.utils import shuffle
    from sklearn.metrics import precision_recall_curve
    import lightgbm as lgb
    import xgboost as xgb
    from itertools import product 
    
    print('permutation-importance'.center(50,'-'))
    print('folds={}, baseline={}'.format(folds,baseline))
    
    def calc_pr(y_true,  probs):
        precisions, recalls ,thrs= precision_recall_curve(y_true, probs)
        mean_precisions = 0.5*(precisions[:-1]+precisions[1:])
        intervals = recalls[:-1] - recalls[1:]
        auc_pr = np.dot(mean_precisions, intervals)
        return auc_pr
        
    feats = data.columns.tolist()
    ret={}
    if isinstance(model, lgb.Booster):
        for col in feats:
            print('feature = {}'.format(col).center(30,'-'))
            mdata = data.copy()
            for _ in range(folds):
                mdata[col] = shuffle(data[col]).tolist()
                y_prob = model.predict(mdata, num_iteration = model.best_iteration)
                aucpr = calc_pr(labels, y_prob)
                ret.setdefault(col,[]).append(aucpr)
                
    elif isinstance(model, xgb.Booster):
        for col in feats:
            mdata = data.copy()
            for _ in range(folds):
                mdata[col] = shuffle(data[col]).tolist()
                dtest = xgb.DMatrix(mdata.values, label =None, weight=None)
                y_prob = model.predict(dtest, ntree_limit = model.best_ntree_limit)
                aucpr = calc_pr(labels, y_prob)
                ret.setdefault(col,[]).append(aucpr)
    else:
        print('permutation-importance ignored')
        return
        
    df = pd.DataFrame(ret)
    df = baseline-df # the more falling the more important
    mean = round(df.mean(),4)
    std = round(df.std(),4)
    view = mean.astype(str)+'(+-'+ std.astype(str)+')'
    df = pd.concat([mean, view],axis=1)
    df.sort_values(by=0,ascending=False,inplace=True)
    df.drop(columns=[0],inplace=True)
    if file:
        df.to_csv(file, index=True)    
    else:
        print(df)

def  calc_one_feature_importance(model_type, train, test, cat_features, imp_file):
    """
    use one feature to train a lgb model, features with auc at some 0.5 are considered unimportant
    """
    assert model_type in ('xgboost, lgboost'), "model_type should be in ('xgboost','lgboost')"
    res ={}
    if model_type == 'xgboost':
        pass
    else:
        params ={
                "learning_rate": 0.08632983726820118,
                "max_depth": 6,
                "num_leaves": 15,
                "min_data_in_leaf": 34,
                "bagging_fraction": 0.8356555454452448,
                "bagging_freq": 1,
                "min_gain_to_split": 2.517518146222459,
                "min_data_in_bin": 56,
                "lambda_l2": 1.6607770905750017,
                "lambda_l1": 3.558364507724445,
                "seed": 42,
                "num_threads": 6,
                "min_sum_hessian_in_leaf": 0.001,
                "max_cat_threshold": 24,
                # "learning_rates": lambda x: 0.002 if x<5 else 0.03*exp(-floor(x/200)),#[0,n_iters)
                "learning_rates": None,
                "objective": "binary",
                "is_unbalance": False,
                'zero_as_missing':False,
                "metric": ["auc"],
                "metric_freq": 5,
                "boosting": "gbdt",
                "verbose": 0,
                'boost_from_average':True  # default True
              } 
        X,y = train[0],train[1]
        for i, col in enumerate(X.columns):
            print('{}/{} ---> {}'.format(i+1, X.columns.size, col))
            Xt = X[[col]]
            cat_feature = [col] if col in cat_features else None
            train_x, val_x, train_y, val_y = train_test_split(Xt, y, stratify=y,  test_size=0.1,random_state=42)
            dtrain = lgb.Dataset(train_x, label = train_y, feature_name='auto',
                        categorical_feature = cat_feature, weight=None)
            dval = lgb.Dataset(val_x, label =val_y,feature_name='auto',
                        categorical_feature = cat_feature, weight=None, reference=dtrain)

            evalists =[dtrain,dval]
            evalnames=['train','val']
            eval_result ={}    
            model = lgb.train(params, dtrain, 50, evalists, evalnames, 
                              early_stopping_rounds=20,
                              verbose_eval = 20, evals_result = eval_result,learning_rates=None)  
            test_x = test[0][[col]]
            test_y = test[1]
            y_prob = model.predict(test_x, num_iteration = model.best_iteration)
            score = roc_auc_score(test_y, y_prob)
            res.update({col:score})
    ser = pd.Series(res)
    ser = ser.sort_values(ascending=False)
    ser.to_csv(imp_file)
    print(ser)

def calc_feature_iv(data, label:str, bins=10, reg=1, filename=None):
    df = pd.read_pickle(data)
    # df = df[df.sec_branch_desc.isin(['江苏','上海','无锡','苏州','南通','浙江','温州','宁波','福建','泉州','厦门'])]
    df = df[df.sec_branch_desc.isin(['北京','天津','青岛','山东'])]
    catcols = df.select_dtypes(include=['category','object']).columns.tolist()
    columns = df.columns.tolist()
    columns.remove(label)
    _gpos = df[label].sum()
    _gneg = df[label].size - _gpos
    iv={}
    for col in columns:
        if col not in catcols:
            tmp = df[col].fillna(-99)
            binv =  pd.cut(tmp,bins=bins,labels=['v'+str(i) for i in range(bins)]) # value of label)
        else:
            binv = df[col].fillna('-99')
        grp = df[label].groupby(binv).agg(['sum',lambda x: x.count()-x.sum()])
        grp = grp.rename(columns={'sum':'pos', '<lambda_0>':'neg'})
        pos = (grp['pos']+reg)/(_gpos+2*reg)
        neg = (grp['neg']+reg)/(_gneg+2*reg)
        iv[col] = ((pos-neg)*np.log(pos/neg)).sum()
        print('{} iv={:.4f}'.format(col, iv[col]))
    df = pd.DataFrame([iv]).T
    df.sort_values(by=0,ascending=False, inplace=True)
    if filename:
        df.to_csv(filename)
    else:
        print(df.head(10))

def calc_psi(file, label:str, tofile=None, time_span=4):
    df = pd.read_pickle(file)

    columns = df.columns.tolist()
    if label in columns: 
        columns.remove(label)
    # app_date = pd.to_datetime(df['app_date'])
    df['app_date'] = df['app_date'].apply(lambda x: x[:7]) #'2018-11-02'
    app_dates = sorted(df['app_date'].unique())[-time_span:][::-1]
    dfs = []
    for date in app_dates:
        print(date)
        tmp = df[df.app_date.str.contains(date)]
        dfs.append(tmp)
    
    total_psi = {}
    # columns =['main_insured_applicant_acc_life_plan_num']
    for col in columns:
        psi = {}
        if df[col].dtype.name in ['object','category']:
            base_cnt = dfs[0][col].value_counts(dropna=False)
            for idx in range(1, len(dfs)):
                cnt = dfs[idx][col].value_counts(dropna=False)
                sub = (base_cnt/base_cnt.sum()).subtract(cnt/cnt.sum(),fill_value=1e-6)
                prod = (base_cnt/base_cnt.sum()+1e-6).divide(cnt/cnt.sum()+1e-6)
                psi.setdefault(col,[]).append((sub*np.log(prod)).sum())
        else:
            base_cnt, bins = pd.cut(dfs[0][col],bins=10,retbins=True)
            base_cnt = base_cnt.value_counts(dropna=False)
            # print(base_cnt)
            for idx in range(1, len(dfs)):
                cnt = pd.cut(dfs[idx][col],bins=bins)
                cnt = cnt.value_counts(dropna=False)
                sub = (base_cnt/base_cnt.sum()).subtract(cnt/cnt.sum(),fill_value=1e-6)
                prod = (base_cnt/base_cnt.sum()+1e-6).divide(cnt/cnt.sum()+1e-6)
                psi.setdefault(col,[]).append((sub*np.log(prod)).sum())
        print(col, psi)
        total_psi.update(psi)
    df = pd.DataFrame(total_psi).T
    df.columns =['N-'+str(i+1) for i in range(time_span-1)]
    df['avg'] = df.mean(axis=1)
    df = df.sort_values(by='avg',ascending=False)
    df.to_csv(tofile)









if __name__ == '__main__':
    import json
    # anotation_path = r'D:\yuanyuqing163\hnb\field_content'
    # iv_file =  r'D:\yuanyuqing163\hnb\iv_40w.csv'
    # with open(anotation_path,'r') as fp:
    #     anotate_dict = json.load(fp)

     
    # df = pd.read_pickle('D:\yuanyuqing163\hnb\data_3w_pickle')
    
    # drop_null_columns(df, null_percent=0.8,
    #                   inplace =True,
    #                   null_columns_store = r'D:\yuanyuqing163\hnb\drop_null_feats.csv',
    #                   anotation_dict =anotate_dict)
    # drop_category_columns(df, category_threshold = 50, 
    #                       inplace=True,
    #                       category_columns_store=r'D:\yuanyuqing163\hnb\drop_category_feats.csv' ,
    #                       anotation_dict=anotate_dict)
    # drop_low_iv_columns(df,iv_file,
    #                     low_iv_store=r'D:\yuanyuqing163\hnb\drop_lowiv_feats.csv' ,
    #                     anotation_dict=anotate_dict)
    data = "/home/yuanyuqing163/hb_model/data/pickle/test_bj_201901_202003_ge18"
    calc_feature_iv(data, 'is_y2', bins=10, reg=1, filename='/home/yuanyuqing163/hb_model/docs/iv_bj.csv')
    # calc_psi(data, 'is_y2',tofile='/home/yuanyuqing163/hb_model/docs/psi_bj.csv')



