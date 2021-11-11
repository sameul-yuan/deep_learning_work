# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:28:19 2019

@author: yuanyuqing163
"""
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import precision_recall_curve

def top_ratio_hit_rate(y_true, y_prob, top_ratio=0.001, sample_threshold= 10):
    """
    calculate the hit-rate of top samples ordered by predicted probability
    
    y_true, y_prob: 1-D array like 
    """
    ns = len(y_true)
    index = np.argsort(y_prob)
    index = index [::-1]
    #top_k
    top_k = int(ns*top_ratio)
    if top_k <= sample_threshold:  # requires at least 10k samples
        top_k = ns
        warnings.warn('--too less samples, total sample is used--')
    index1 = index[:top_k]
    top_true =  np.array(y_true)[index1] 
    hit_rate = sum(top_true)/top_k

    top30_k = int(0.3*ns)
    index2 = index[:top30_k]
    top30_true = np.array(y_true)[index2]
    cover = sum(top30_true)/sum(y_true)


    return hit_rate, top_k, index, cover
    
# custom-metric for xgb_model
def feval_top_hit(y_prob, dtrain):
    y_true = dtrain.get_label()
    hit_rate, _= top_ratio_hit_rate(y_true, y_prob)
    return 'hit_rate', hit_rate
    
# custom-metric for lgb_model   
def feval_top_hit_lgb(y_pred, dtrain):
    y_true = dtrain.get_label()
    hit_rate, _ = top_ratio_hit_rate(y_true, y_pred) 
    return "rate", hit_rate, True  #is_larger_better

def feval_aucpr_lgb(y_pred, dtrain):
    y_true = dtrain.get_label()
    precisions, recalls ,thrs = precision_recall_curve(y_true, y_pred)
    mean_precisions = 0.5*(precisions[:-1]+precisions[1:])
    intervals = recalls[:-1] - recalls[1:]
    auc_pr = np.dot(mean_precisions, intervals)
    return 'aucpr', auc_pr, True 

def eval_gini_normalization(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    index = np.argsort(y_pred) #min->max
    y_true = y_true[index]
    cumsum = np.cumsum(y_true)
    ratios = cumsum/sum(y_true)
    ratios = (ratios[:-1]+ratios[1:])/2
    auc_gini = sum(ratios)/(len(ratios))
    return 0.5 - auc_gini
    
def eval_aucpr(y_true, y_pred):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    precisions, recalls ,thrs = precision_recall_curve(y_true, y_pred)
    mean_precisions = 0.5*(precisions[:-1]+precisions[1:])
    intervals = recalls[:-1] - recalls[1:]
    auc_pr = np.dot(mean_precisions, intervals)
    return auc_pr
    


def label_encoder(df, cols, na_sentinel=-1):
    '''
    numeric encoding (label encoding)for series and 1-D list, inplace operation
    since df object is mutable.
    
    na_sentinel:  mark  null value or na
    pd.factorize(['a','b','c'])->[0,1,2]
    '''
    columns = df.columns
    for col in cols:
        if col in columns:
            df[col] = pd.factorize(df[col].values, na_sentinel=na_sentinel)[0]  
            
def cast_to_numeric(df, cols, errors='raise', downcast='integer'):
    """
    convert df columns to numeric dtype, 
    """
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors, downcast)
    
def cut_columns(df, cols, bins=8, **kw):
    """
    discrete continous variables into equal-width buckets,  inplace operation  
    
    kw: dict-params for pd.cut
    """
    bin_labels = list(map(lambda x: 'C'+str(x), range(bins)))
    for col in cols:
        df[col] = pd.cut(df[col], bins, labels=bin_labels, **kw)

def qcut_columns(df, cols, q=4, **kw):
    """
    discrete continous variables into equal-freq buckets , inplace operation  
    
    kw : dict-param for pd.qcut
    """
    bin_labels = list(map(lambda x:'C'+str(x), range(q)))
    for col in cols:
        df[col]=pd.qcut(df[col],q, labels=bin_labels, **kw)
    

def get_numeric_cols(df, includes=('int,float')):
	cols = []
	func = lambda x: any((map(lambda y: x.startswith(y),includes)))
	for col in df.columns:
		if func(df[col].dtype.name):
			cols += [col]
	return cols
		
def safe_cast_float2int_by_fillna(df, cols:list, replace_na:int =-99,
                                  downcast='integer'): 
    """
    ====
    examples: 
      [1.0, 2.0, 3.0] ->[1,2,3]
      [1.0, 2.0, np.nan] ->[1, 2, replace_na]
      [1.1, 2.0, 3.0] -> [1.1, 2.0, 3.0], do nothing
      [1.1, 2.0, np.nan] ->[1.1, 2.0, np.nan],  do nothing
    """
    success_cols =[]
    for col in cols:
        # if df[col].isnull().sum()>0 # float type due to numeric + nan
        if col in df.columns:
            # print(f'----{col}----')
            tmp = df[col].fillna(replace_na)
            cast = pd.to_numeric(tmp, downcast=downcast)
            if cast.dtype.name.startswith('int'):# successful convert
                df[col] = cast.astype(str) 
                success_cols +=[col]
    return success_cols
        
        


def count_category_number(df, *, includes=('object', 'category')):
    """
    panda dtypes include 'object, category, int ,float'
    
    return value categories of columns  with dtype included 
    in param'includes'.
    """
    cols = df.columns
    ret = pd.Series()
    func = lambda x: any((map(lambda y: x.startswith(y),includes)))
    for col in cols:
        if func(df[col].dtype.name):  #column dtype satisfied
            ret = ret.append(pd.Series(df[col].unique().size,index=[col]))
            
    return ret

def create_feature_map(features, outfile):
#    assert isinstance(features,(list,tuple)),' param feature need by list/tuple'
    with open(outfile,'w+') as fp:
        for i, feat in enumerate(features):
            fp.write('{0}\t{1}\tq\n'.format(i, feat))
  
def save_xgb_model(xgb_model, model_path, features=None, fmap=None, importance_type='weight',imp_path=None):
    import operator
    xgb_model.save_model(model_path)    
    
    if imp_path is not None and fmap is not None:
        if features is not None:
            create_feature_map(features, fmap)
        importance=xgb_model.get_score(fmap=fmap,importance_type=importance_type) 
        importance = sorted(importance.items(), key=operator.itemgetter(1),reverse=True)
        df = pd.DataFrame(importance,columns=['feature', 'fscore'])
        df.set_index('feature',inplace=True)
        df['fscore'] = df['fscore']/df['fscore'].sum()
        df.to_csv(imp_path)
        
        xgb_model.dump_model(model_path+'_raw', fmap=fmap)

def save_lgb_model(lgb_model, model_path, features=None, cat_features=None, imp_type ='gain',imp_path =None):
    import operator
    import os
    lgb_model.save_model(model_path)
    if cat_features is not None and features is not None:
        feature_path = os.path.split(model_path)[0] + '/lgb_features.csv'
        feature_dict = {}
        feature_dict['feature'] = features
        for feature in features:
            type_ = 'string' if feature in cat_features else 'double'
            feature_dict.setdefault('-',[]).append(type_)
            feature_dict.setdefault('--',[]).append('optional')
        df = pd.DataFrame(feature_dict)
        df.to_csv(feature_path,index=False)       
    if imp_path is not None:
        importance = lgb_model.feature_importance(importance_type=imp_type, iteration=lgb_model.best_iteration)
        df = pd.DataFrame(importance, index=features, columns=['fscore'])
        df['fscore'] = df['fscore']/df['fscore'].sum()
        df.sort_values(by='fscore',ascending=False,inplace=True)
        df.index.name='feature'
        df.to_csv(imp_path)
        
            
def calc_threshold_vs_depth(y_true, y_prob, stats_file=None):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    # print(y_prob[:100])
    ns = len(y_true)
    index = np.argsort(y_prob)
    index = index[::-1]
    y_prob = y_prob[index]
    # print(y_prob[:100])
    ratios = [0.001,0.002,0.003,0.004,0.005, 0.01,0.05, 0.1,0.15, 0.2, 0.25,0.3,
               0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,1]
    
    pos_num = sum(y_true)
    pv = pos_num/len(y_true)
    print('pos-to-neg-ratio=%f'%pv)
    depths =[]
    rates =[]
    samples=[]
    covers =[]
    lifts=[]
    p_thresholds=[]
    for ratio in ratios:
        top_k = max(1,int(ns*ratio))
        index1 = index[:top_k]
        top_true =  y_true[index1] 
        hit_rate = sum(top_true)/top_k  
        cover = sum(top_true)/pos_num
        p_threshold = y_prob[top_k-1]
        lift = hit_rate/pv
        
        depths.append(ratio)
        rates.append(hit_rate)
        samples.append(top_k)
        covers.append(cover)
        lifts.append(lift)
        p_thresholds.append(p_threshold)
        
    df = pd.DataFrame({'深度':depths,'命中率': rates, '覆盖率':covers, '样本数':samples,
                  '提升度':lifts, '概率门限':p_thresholds})
    if stats_file is not None:
        df.to_csv(stats_file, encoding='gbk')   
    print(df)
    # return p_thresholds[0], p_thresholds[11], p_thresholds[15], p_thresholds[20]
    return rates[0], rates[4], covers[11], covers[12]

def model_merge_stats(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    # print(y_prob[:100])
    ns = len(y_true)
    index = np.argsort(y_prob)
    index = index[::-1]
    y_prob = y_prob[index]
    # print(y_prob[:100])
    ratios = [0.001,0.005, 0.3]
    
    pos_num = sum(y_true)
    pv = pos_num/len(y_true)
    print('pos-to-neg-ratio=%f'%pv)
    depths =[]
    rates =[]
    samples=[]
    covers =[]
    lifts=[]
    p_thresholds=[]
    for ratio in ratios:
        top_k = max(1,int(ns*ratio))
        index1 = index[:top_k]
        top_true =  y_true[index1] 
        hit_rate = sum(top_true)/top_k  
        cover = sum(top_true)/pos_num
        p_threshold = y_prob[top_k-1]
        lift = hit_rate/pv
        
        depths.append(ratio)
        rates.append(hit_rate)
        samples.append(top_k)
        covers.append(cover)
        lifts.append(lift)
        p_thresholds.append(p_threshold)
        
    df = pd.DataFrame({'深度':depths,'命中率': rates, '覆盖率':covers, '样本数':samples,
                  '提升度':lifts, '概率门限':p_thresholds})  
    # print(df)
    return rates[0], rates[1], covers[2]

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('------------\nmemory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
    
        
        
def topological_sort(l1:list, l2:list)->list:
    assert isinstance(l1, list) and isinstance(l2, list), 'input should be list'
    comm = sorted(set(l1).intersection(l2), key = l1.index)
    ret =[]
    prev1 = prev2= -1
    for e in comm:
        cur1 = l1.index(e)
        cur2 = l2.index(e)
        ret.extend(l1[prev1+1:cur1]+l2[prev2+1:cur2]+[e])
        prev1,prev2 = cur1, cur2
    ret.extend(l1[prev1+1:]+l2[prev2+1:])
    return ret    
    
	
class MapBinCDF(object):
    def __init__(self, number_bins=100):
        self.num_bins = number_bins
        self.bins = 0
        self.bin_means = {}
        self.min_v = np.finfo('float32').min
        self.max_v = np.finfo('float32').max
    
    def fit(self, X: pd.Series):
         # _, bins = pd.qcut(X.values, q=self.num_bins, retbins=True, duplicates='drop') #
        X = np.array(sorted(X.values)) # list
        self.min_v, self.max_v = X[0], X[-1]
        nums = len(X)
        inds = list((nums*np.arange(0,self.num_bins)/(self.num_bins)).astype(int))
        inds.append(nums-1) #最后一个数
        
        inds_reduce = []
        bin_edges= []
        inds_reduce.append(0)
        bin_edges.append(X[0])
        prev = X[0]
        
        for ind in inds[1:]:  #drop duplicates
            if X[ind]==prev:
                continue
            else:
                inds_reduce.append(ind)
                bin_edges.append(X[ind])
                prev = X[ind]

        for key, (s, e) in enumerate(zip(inds_reduce[:-1],inds_reduce[1:])):
            if key!=len(inds_reduce[:-1])-1:
                self.bin_means[key]=X[s:e].mean()
            else:
                self.bin_means[key]=X[s:e+1].mean()
                
        self.bins = bin_edges
        
        self.key_min ,self.key_max = min(self.bin_means.keys()),max(self.bin_means.keys())
            
    def transform(self, X: pd.Series):
        X = X.map(lambda x: self.min_v if x<=self.min_v else x)
        X = X.map(lambda x: self.max_v if x>=self.max_v else x)
        index = np.searchsorted(self.bins, X, side='right') #`a[i-1] <= v < a[i]
        index = index-1
        index = np.asarray([min(max(self.key_min, v),self.key_max) for v in index])
        return index/(len(self.bins)-1)  #cdf 
        
    def invert_transform(self, X:np.array):
        # X Predicted quantiles
#         X = map(lambda x:  self.min_p if x<=self.min_p else x, X)
#         X = map(lambda x:  self.max_p if x>=self.max_p else x, X)
#         X = (int(x*len(self.bins)) for x in X)
        X =  [int(x*(len(self.bins)-1)) for x in X]
        X =  [min(max(self.key_min,x),self.key_max) for x in X]
        res = np.asarray([self.bin_means[x] for x in X])
        return res     
    
    
    
