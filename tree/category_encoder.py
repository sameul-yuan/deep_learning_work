
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:34:14 2019

@author: yuanyuqing163
"""

import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
import numpy as np
import pickle
import six
from scipy.sparse import csr_matrix 
from itertools import cycle
from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin

#---------------------------------------------------------------
class DiscreteMixin(object):
    def get_encode_cols(self, df):
        dtype_to_encode=['object','category']
        cols = df.select_dtypes(include=dtype_to_encode).columns.tolist()
        return cols

    def save(self, encoder_file:str):
        with open(encoder_file,'wb') as f:
            pickle.dump(self.__dict__, f)
            
    def load(self,encoder_file:str):
        with open(encoder_file,'rb') as f:
            self.__dict__.update(pickle.load(f))

class custom_json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        elif isinstance(obj, datetime.date):
            return obj.strftime('%Y/%m/%d')
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        else:
            return super(custom_json_serialize, self).default(obj)
        
class Normalize(BaseEstimator, TransformerMixin, DiscreteMixin):
    """
    normalize to N(0,1) by z-score or [0-1] by min-max
    """
    def __init__(self, skipna=None, ddof=1, type='z-score', epsilon= 1e-8, **kwargs):

        self.skipna = skipna
        self.ddof = ddof
        self.type = type
        self.normal_cols = []
        self.scaler ={}
        self.epsilon = epsilon
        self.kwargs = kwargs
        assert self.type in ('z-score','min-max'), "type support only ('z-score', 'min-max')"

    @staticmethod
    def get_encode_cols(df, dtype_to_encode=None):
        if dtype_to_encode is None:
            dtype_to_encode =[np.number]
        cols = df.select_dtypes(include=dtype_to_encode).columns.tolist()
        return cols

    def fit(self, X, y=None, includes=None, excludes=None):
        """
        parameter
        ----------
        X: DataFrame  to generate one-hot-encoder rule
		y: label column in DataFrame X if provided

        includes: specify the columns  to be  normalized
        excludes: if 'includes' is provided this param will
           not be used, otherwise all numeric columns except 'excludes'
           will be normalized.
        """
        print('{}.{}...'.format(self.__class__.__name__, get_current_function()))
        assert isinstance(X, DataFrame), 'X should be DataFrame object'
        columns = X.columns.tolist()
        if y is not None:
            if y not in columns:
                raise ValueError('y is not in X.columns during {}.fit'.format(self.__class__.__name__))
            else:
                columns.remove(y)

        # get encoder columns
        if includes is None:
            cols = self.get_encode_cols(X)
            if excludes is not None:
                cols = [col for col in cols if col not in excludes]
        else:
            assert excludes is None, 'excludes should be None if includes provided'
            cols = includes
        if y in cols:
            cols.remove(y)
        # re-order cols by original order
        cols = sorted(list(set(cols)), key= columns.index)

        scaler ={}
        scaler['mean'] = X[cols].mean(skipna=self.skipna)
        scaler['std'] = X[cols].std(ddof=self.ddof)
        if self.type =='min-max':
            scaler['min'] = X[cols].min()
            scaler['max'] = X[cols].max()
        self.normal_cols = cols
        self.scaler = scaler

        return self

    def transform(self, X, y=None, dtype=None, inplace=False):
        """
        parameter
        -----------
        dtype: specifies the dtype of encoded value
        """
        print('{}.{}...'.format(self.__class__.__name__, get_current_function()))
        assert isinstance(X, DataFrame), 'X shoule be DataFrame object'
        columns = X.columns.tolist()
        if y is not None:
            if y not in columns:
                raise ValueError("'y label {}' not in X".format(y))
            else:
                columns.remove(y)

        diffs = set(self.normal_cols).difference(columns)
        if len(diffs) > 0:
            raise ValueError("X not includes double columns '{}'".format(diffs))

        if not inplace:
            X = X.copy()  # X=X.copy(deep=True)  default_fill_value

        X[self.normal_cols] = X[self.normal_cols].fillna(self.scaler['mean'].to_dict())
        if isinstance(X, pd.SparseDataFrame):
            X._default_fill_value = np.nan
        if self.type=='z-score':
            X[self.normal_cols] = (X[self.normal_cols]-self.scaler['mean'])/(self.scaler['std'] + self.epsilon)
        else:
            lb = self.kwargs.get('lbound',0)
            hb = self.kwargs.get('hbound',1)
            X[self.normal_cols] =  lb + (X[self.normal_cols] - self.scaler['min'])/(
                self.scaler['max']-self.scaler['min'])*(hb-lb)
            X[self.normal_cols] = X[self.normal_cols].applymap(lambda x: min(max(x,lb),hb))


        return X

                
class TargetEncoder(BaseEstimator, TransformerMixin, DiscreteMixin):
    def __init__(self, handle_missing='encoder',  handle_unknown='mean',
                 min_samples_leaf=1, smoothing=1):
        """Target Encode for categorical features. Based on leave one out approach,
           and designed for binary classification.
         
        Parameters
        ----------
        handle_missing: str
            options are 'encoder','coerce','mean', default to 'mean' which fill missing value with mean value. 
            'encoder' treats missing value as a category, and 'coerce' keep it as np.nan
        handle_unknown: str
            options are 'error', 'coerce' and 'mean', defaults to 'mean',  and 'coerce' keep it as np.nan.
        min_samples_leaf : int
            minimum samples to take category average into account
        smoothing : int
            smoothing effect to balance categorical average vs prior

        Example
        -------
        >>>from category_encoders import *
        >>>import pandas as pd
        >>>from sklearn.datasets import load_boston
        >>>bunch = load_boston()
        >>>y = bunch.target
        >>>X = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        >>>enc = TargetEncode(cols=['CHAS', 'RAD']).fit(X, y)
        >>>numeric_dataset = enc.transform(X)
        """
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing
        self.handle_missing = handle_missing
        self.handle_unknown = handle_unknown
        self._mean = None
        self._dim = None
        self.mapping = {}
        self.encode_cols = []
        
        if self.handle_missing  not in('encoder','coerce','mean'):
            raise ValueError('handle_missing not in(encoder,coerce,mean)')
        if self.handle_unknown not in('error','coerce','mean'):
            raise ValueError('handle_unknown not in(error,coerce,mean)')
        
    def fit(self, X, y, cols_to_encode=None, extra_numeric_cols=None):
        """Fit encoder according to X and y.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        Returns
        -------
        self : encoder
            Returns self.
        """
        # first check the type
#        X = convert_input(X)
        assert isinstance(X, DataFrame), ' X should be DataFrame type'
        if isinstance(y, six.string_types) and y in X.columns:
            y = pd.Series(X[y].values, name='target')
            self._dim = X.shape[1] -1
        else:
            assert X.shape[0] == y.shape[0]
            y = pd.Series(y, name='target')
            self._dim = X.shape[1]
          
        # if columns aren't passed, just use every string column
        if cols_to_encode is None:
            self.encode_cols = self.get_encode_cols(X)
            self.encode_cols += list(extra_numeric_cols) if extra_numeric_cols is not None \
                               else []
        else:
            self.encode_cols = cols_to_encode
        print('{} columns will be encoded'.format(len(self.encode_cols)))
        self.mapping = {}
        self._mean = y.mean()  # priori information
        
        n_critical = self.min_samples_leaf
        k_factor = self.smoothing
        
        for col in self.encode_cols:
            tmp = y.groupby(X[col]).agg(['sum', 'count'])
            tmp['mean'] = tmp['sum'] / tmp['count']
            tmp = tmp.to_dict(orient='index')
            for val in tmp:
                if tmp[val]['count'] == 1:
                    target_val = self._mean
                else:
                    smoothing = 1 / (1 + np.exp(-(tmp[val]["count"] - n_critical) / k_factor))
                    target_val = self._mean * (1 - smoothing) + tmp[val]['mean'] * smoothing
                tmp[val]['smoothing'] = target_val
 
            if self.handle_missing =='mean':
                fillna = self._mean
            elif self.handle_missing =='encoder':
                missing = X[X[col].isnull()][col]
                target = y[missing.index]
                count = target.index.size
                mean = target.sum()/count if count>=1 else 0
                if count<=1:
                    fillna = self._mean
                else:
                    smoothing = 1 / (1 + np.exp(-(count- n_critical) / k_factor))
                    fillna = self._mean*(1-smoothing) + mean*smoothing
                tmp['null'] ={'sum':target.sum(), 'count':count, 'mean':mean,
                               'smoothing':fillna}
            else:
                pass  # coerce do  nothing            
            self.mapping[col] = tmp           
            
        return self


    def transform(self, X, y=None, inplace=False, return_df=True):
        """Perform the transformation to new categorical data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples] when transform by leave one out
            None, when transform withour target infor(such as transform test set)
            
        Returns
        -------
        p : array, shape = [n_samples, n_numeric + N]
            Transformed values with encoding applied.
        """

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # first check the type
#        X = convert_input(X)
        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))
        assert (y is None or X.shape[0] == y.shape[0])

        Xc = X if inplace else X.copy()
        
        if not self.encode_cols:
            return Xc
        
        if self.handle_unknown=='coerce':
            filluk = np.nan
        elif self.handle_unknown =='mean':
            filluk = self._mean
      
        for col_ in self.encode_cols: # {'col':col,'mapping':mapping}
            maps = self.mapping.get(col_)
            vals = Xc[col_].dropna().unique()
            tmp = Xc[col_].copy()
            
            #handle missing value      
            if self.handle_missing == 'mean':
                fillna = self._mean  #X[switch.get('col')].fillna(self._mean, inplace=True)
            elif self.handle_missing == 'encoder':
                fillna = maps['null']['smoothing']
            else:
                fillna = np.nan
            tmp.fillna(fillna,inplace=True)
            
            #handle unknown value, excluding na
            diff = set(vals).difference(set(maps.keys()))
            if len(diff) != 0: # some element only appers in test set
                print("unseen {}  in column'{}".format(diff,col_))
                tmp.loc[Xc[col_].isin(diff)] = filluk
                vals = set(vals).difference(diff)  
                
            for val in vals:                
                if maps[val]['count'] == 1:
                    tmp.loc[Xc[col_] == val] = self._mean
                else:
                    tmp.loc[Xc[col_] == val] = maps[val]['smoothing']                            

            Xc.loc[:, col_] = tmp.astype(float)
            
        return Xc if return_df else Xc.values

           
class OneHotEncoder(BaseEstimator, TransformerMixin, DiscreteMixin):  
    
    def __init__(self, dummy_na=True, handle_unknown='ignore',
                 category_threshold=50, drop_threshold_cols=True,
                 replace_na=-99):
        """
        parameter
        ---------
        dummy_na: bool, defualt True
        handle_unknown: str, 'error' or 'ignore'
        category_threshold: columns of categories more then this threhold will
                            not be encoded
        drop_threshold_cols: drop columns that not satisfy category_threshold 
                             or columns of one category
        """
        self.dummy_na = dummy_na
        self.handle_unknown = handle_unknown
        self.category_threshold = category_threshold
        self.drop_threshold_cols = drop_threshold_cols
        self.encode_cols= []
        self.drop_cols=[]
        self.mapping = {}
        self.replace_na = replace_na
        self._dim = None
        
    def fit(self, X, y=None, cols_to_encode=None, extra_numeric_cols=None):
        """
        parameter
        ----------
        X: DataFrame obj to generate one-hot-encoder rule
		y: label name in DataFrame X if provided
        
        cols_to_encoders: specify the columns  to be  encoded
        extra_numeric_cols: if cols_to_encoder is provided this param will
           not be used, otherwise all object columns and extra_numeric_cols 
           will be encoded.
        """
        assert isinstance(X, DataFrame),'X should be DataFrame object'
        if y is not None:
             if y not in X.columns:
                raise ValueError('y is not in X.columns during fit')
             else:
                self._dim = X.shape[1]-1
        else:
            self._dim = X.shape[1]
        # get encoder columns 
        if cols_to_encode is None:
            cols = self.get_encode_cols(X)
            cols += list(extra_numeric_cols) if extra_numeric_cols is not None \
                    else []
        else:
            cols = cols_to_encode   
        cols = list(set(cols))
        if len(cols)==0:
            return
    
        if y in cols:
            cols.remove(y)
        # re-order cols by original order 
        cols = sorted(cols, key=X.columns.get_loc)       
        # convert na to pre-defined value
        df = X[cols].fillna(self.replace_na,downcast='infer')
        
        # generate rules 
        cats_list = pd.Series()
        for col in cols:
            cats_list[col] = df[col].unique().tolist()
            if (not self.dummy_na) and (self.replace_na in cats_list[col]):
                cats_list[col].remove(self.replace_na)       
        cats_cnt = cats_list.apply(lambda x: len(x))
        # exclude columns of too manay categories or just one category
        drop_mask = (cats_cnt > self.category_threshold) | (cats_cnt==1)
        drop_index = cats_cnt[drop_mask].index
        cats_list = cats_list[~cats_list.index.isin(drop_index)]
        
        self.drop_cols = drop_index.tolist()
        self.encode_cols = cats_list.index.tolist()
        maps={}
        for col in self.encode_cols:
            # map each val in col into a index
            val_list = cats_list[col]   
            val_map = OrderedDict({val:i for i,val in enumerate(val_list)})
            maps[col] = val_map
        self.mapping = maps
        
        return self
  
    def transform(self, X, y=None, dtype=None, inplace=False): 
        """
        parameter
        -----------
        dtype: specifies the dtype of encoded value
        """
        assert isinstance(X, DataFrame),'X shoule be DataFrame object'
        if  y is not None:
            if y not in X.columns:
                raise ValueError("'y label {}' not in X".format(y))
            assert self._dim == X.shape[1] -1 
        else:
            assert self._dim == X.shape[1]        
            
        diff_cols = set(self.encode_cols).difference(set(X.columns))
        if len(diff_cols)>0:
            raise ValueError("X not includes encoded columns '{}'".format(diff_cols))
        
        if not inplace:
            X = X.copy() # X=X.copy(deep=True)
            
        if self.drop_threshold_cols:
            X.drop(self.drop_cols,axis=1, inplace=True)
            
        data_to_encode = X[self.encode_cols].fillna(self.replace_na,
                        downcast='infer')
        with_dummies = [X.drop(self.encode_cols,axis=1)]
        
        prefix = self.encode_cols
        prefix_sep = cycle(['_'])
        
        for (col, pre, sep) in zip(data_to_encode.iteritems(), prefix,
                                   prefix_sep):
            # col is (col_name, col_series) type
            dummy = self._encoder_column(col[1], pre, sep, dtype = dtype)
            with_dummies.append(dummy)
            
        result = pd.concat(with_dummies, axis=1)
        
        return result
      
    def _encoder_column(self, data, prefix, prefix_sep, dtype):
        
        if dtype is None:
            dtype = np.uint8
            
        maps = self.mapping[prefix]
        dummy_strs = cycle([u'{prefix}{sep}{val}'])
        dummy_cols = [dummy_str.format(prefix=prefix,sep=prefix_sep,val=str(v))
                      for dummy_str, v in zip(dummy_strs, maps.keys())]
        
        if isinstance(data, Series):
            index = data.index
        else:
            index = None
            
        row_idxs= []
        col_idxs= []
        for i, v in enumerate(data):
            idx = maps.get(v,None)
            if idx is None:
                print("{} only exist in test column '{}'".format(v, prefix))
            else:
                row_idxs.append(i)
                col_idxs.append(idx)
        sarr = csr_matrix((np.ones(len(row_idxs)),(row_idxs,col_idxs)),shape=
                          (len(data),len(dummy_cols)), dtype=dtype)
        out = pd.SparseDataFrame(sarr, index=index, columns=dummy_cols,
                           default_fill_value=0,dtype=dtype)
        return out.astype(dtype)  # care of row and columns not covered by sarr
 
                
class LabelEncoder(BaseEstimator, TransformerMixin, DiscreteMixin):
      
    def __init__(self, dummy_na=True,  handle_unknown='coerce',replace_na= -99):
        """
    Parameters
    ----------
    dummy_na: bool, force to  True
    handle_unknown: str
        options are 'error', 'coerce', default to 'coere',  and 'coerce' keep it as replace_uk
         unknown value maps to the biggest integer in the mapping dictionary
        """
        self.dummy_na = dummy_na
        self.handle_unknown = handle_unknown
        self.replace_na = replace_na       
        self._dim = None
        self.mapping = {}
        self.encode_cols = []
        
        if not dummy_na:
            raise ValueError('dummy_na not support take value of False')
        if self.handle_unknown not in('error','coerce'):
            raise ValueError('handle_unknown not in(error,coerce)')
        
    def fit(self, X, y=None, cols_to_encode=None, extra_numeric_cols=None):
        """Fit encoder according to X and y.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : str, default None
            Target name in X if exsit.
        cols_to_encode : columns need LabelEncode
        extra_numeric_cols: will not be used if cols_to_encode is provided, 
            otherwise, extra_numeric_cols combines with other 'object' column 
            will be encoded.
        Returns
        -------
        self : encoder
            Returns self.
        """
        # first check the type
#        X = convert_input(X)
        assert isinstance(X, DataFrame), ' X should be DataFrame type'
        if y is not None and y in X.columns:
            remove_y = y
            self._dim = X.shape[1]-1
        else:
            remove_y =[]
            self._dim = X.shape[1]
        # if columns aren't passed, just use every string column
        if cols_to_encode is None:
            self.encode_cols = self.get_encode_cols(X)
            if extra_numeric_cols is not None:
                self.encode_cols += list(extra_numeric_cols)
            if remove_y in self.encode_cols:
                self.encode_cols.remove(remove_y)
        else:
            self.encode_cols = cols_to_encode
        self.encode_cols = list(sorted(set(self.encode_cols), key=self.encode_cols.index))
        print('{} columns will be encoded'.format(len(self.encode_cols)))
        # convert na to pre-defined
        if self.dummy_na:
            df = X[self.encode_cols].fillna(self.replace_na, downcast='infer').astype(str)
        for col in self.encode_cols:
            vals = df[col].sort_values().unique()
            # if str(self.replace_na) in vals:
                # vals.remove(str(self.replace_na))  # map na  to -1 
                # maps ={val: idx for idx,val in enumerate(vals)}
                # maps[str(self.replace_na)] = -1
            # else:
                # maps ={val: idx for idx,val in enumerate(vals)}
            maps ={val: idx for idx,val in enumerate(vals)}
            maps['uk'] = max(maps.values())+1
            # maps['uk'] = -1
            self.mapping[col] = maps
        return self
        
    def transform(self, X, y=None, inplace=False, return_df=True):
        """Perform the transformation to new categorical data.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        y : str or None
        Returns
        -------
        p : array, shape = [n_samples, n_features]
            Transformed values with encoding applied.
        """
        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')
        # then make sure that it is the right size
        if y is not None and y in X.columns:
            assert X.shape[1] == self._dim -1
        else:
            assert X.shape[1] == self._dim

        Xc = X if inplace else X.copy(deep=True)
        if not self.encode_cols:
            return Xc
        
        if self.dummy_na:
            Xc[self.encode_cols] = Xc[self.encode_cols].fillna(self.replace_na, downcast='infer').astype(str)
        
        msg = "unseen category {} in column '{}'"
        for col in self.encode_cols:
            maps = self.mapping[col]
            tmp = Xc[col]
            tmp = Xc[col].map(maps)  # unseen value maps to NaN 
            unseen = Xc.loc[tmp.isnull(), col]
            if not unseen.empty:
                if self.handle_unknown =='error':
                    raise ValueError(msg.format(set(unseen.values), col))
                else:
                    print(msg.format(set(unseen.values), col))
                    tmp = tmp.fillna(maps['uk'],downcast='infer')
            Xc[col] = tmp.astype(int)  
            
        return Xc if return_df else Xc.values
               
class MeanEncoder(BaseEstimator, TransformerMixin, DiscreteMixin):
    def __init__(self, dummy_na = True, handle_unknown='prior', n_critical=1, 
                 scale_factor=1, drop_last = False, replace_na= -99):
        """
        dummy_na: bool,if False the null values will be repaced with prior after
                  transform
        handle_unknown: str, 'error' of 'prior'
        drop_last: bool,whether to get C-1 categories out of C by removing the
                   last class.
        n_critical: the critical point that the posterior will contribute more
        scale_factor: scale the smoothing factor
        replace_na : int
        """
        self.dummy_na = dummy_na
        self.handle_unknown =handle_unknown
        self.n_critical = n_critical
        self.scale_factor = scale_factor
        self.drop_last = drop_last
        self.mapping={}
        self.prior=None
        self.encode_cols= None
        self.replace_na = replace_na  # 
        self._dim =None     # attribution dimension
        
    def fit(self, X, y, cols_to_encode=None, extra_numeric_cols=None):
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X should be DataFrame type')
        
        if isinstance(y, six.string_types):
            if y in X.columns:
                self._dim = X.shape[1] - 1
                X = X.rename(columns={y:'_y_'})
            else:
                raise ValueError('y not in X.columns during fit')
        else:
            self._dim = X.shape[1]
            y = pd.Series(y, name='_y_')
            X = X.join(y)
            
        X['_y_'] = X['_y_'].astype(int)   
        
        # get encoder columns 
        if cols_to_encode is None:
            cols = self.get_encode_cols(X)
            cols += list(extra_numeric_cols) if extra_numeric_cols is not None\
                    else []
        else:
            cols = cols_to_encode    
        cols = list(set(cols))
        
        if len(cols)==0:
            return self
        # re-order cols by original order 
        cols = sorted(cols, key=X.columns.get_loc)
        self.encode_cols = cols
        
        data_to_encode = X[self.encode_cols+['_y_']]
        # convert na to a pre-defined value 
        data_to_encode.fillna(self.replace_na, downcast='infer',inplace=True)
        prior = data_to_encode['_y_'].value_counts()/len(data_to_encode['_y_'])
        prior.sort_index(axis=0,inplace=True)
        prior.name='prior'
        self.prior = prior  #series
        
        maps = {}
        for col in self.encode_cols:
            ctb = pd.crosstab(index=data_to_encode[col], columns=data_to_encode['_y_'])
            # deal with missing y.
            ctb = ctb.reindex(columns=prior.index, fill_value = 0)
            ctb.sort_index(axis=1,inplace=True)
            # calculate posterior 
            post = ctb.apply(lambda x: x/x.sum(), axis =1)
            # calcalate smoothing factor of prior and posterior
            smooth = ctb.applymap(lambda x: 1/(1+np.exp(-(x-self.n_critical)/self.scale_factor)))
            smooth_prior = (1-smooth).multiply(prior,axis=1) # DataFrame multiple series
            smooth_post =  smooth.multiply(post)
            codes = smooth_prior + smooth_post
            # normalize
            codes = codes.divide(codes.sum(axis=1),axis=0)
            # encode na with prior if na is not treated as a cateogry
            if not self.dummy_na and self.replace_na in codes.index:
                codes.loc[self.replace_na,:]=self.prior
            maps[col] =codes
            
        self.mapping = maps
        
        return self
            
    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X should be DataFrame type')
        if isinstance(y, six.string_types) and y in X.columns:
            if  self._dim != X.shape[1] -1:
                raise ValueError('dimension error')
        elif self._dim != X.shape[1]:
            raise ValueError('dimension error')
        
        if not self.encode_cols:
            return X
        data_to_encode = X[self.encode_cols]
        #fill na 
        data_to_encode.fillna(self.replace_na, downcast='infer',inplace=True)
        with_dummies = [X.drop(self.encode_cols,axis=1)]
        
        prefix = self.encode_cols
        prefix_sep = cycle(['_'])
        
        for (col, pre, sep) in zip(data_to_encode.iteritems(), prefix,
                                   prefix_sep):
            # col is (col_name, col_series) type
            dummy = self._encode_column(col[1], pre, sep)
            with_dummies.append(dummy)

        result = pd.concat(with_dummies, axis=1)
        
        return result
    
    def _encode_column(self, data, prefix, prefix_sep):
              
        maps = self.mapping[prefix]
        dummy_strs = cycle([u'{prefix}{sep}{val}'])
        dummy_cols = [dummy_str.format(prefix=prefix,sep=prefix_sep,val=str(v))
                      for dummy_str, v in zip(dummy_strs, maps.columns)]
        
        if isinstance(data, pd.Series):
            index = data.index
        else:
            index = None
        
        enc_df = maps.loc[data.values,:] # NaN with unknonw value
        #handle unknown value
        if not all(data.isin(maps.index)):
            msg = "unknown category {} in column '{}'".format(
                        data[~data.isin(maps.index)].values, prefix)
            if self.handle_unknown=='error' :
                raise ValueError(msg)
            else:
                print(msg)
                enc_df.fillna(self.prior, inplace=True)  
        enc_df.index = index
        enc_df.columns = dummy_cols
        if self.drop_last:
            enc_df = enc_df.iloc[:,:-1]
            
        return enc_df
        

class WoeEncoder(BaseEstimator, TransformerMixin, DiscreteMixin):
    """
     currently only support discrete variable encode.
    """
    def __init__(self, dummy_na = True, handle_unknown='coerce', replace_na=-99,
                 reg = 1):
        '''
        dummy_na: bool, if true null value is treated as a category, otherwise
                  null value will be filled with zero.              
        handle_unknown: one of ('zero', 'error','coerce')
        reg: int, bayesian prior value  to avoid divding by zero when calculate woe.
        
        '''
        # super(WoeEncoder, self).__init__()
        self.dummy_na = dummy_na
        self.handle_unknown = handle_unknown
        self.replace_na = replace_na
        self.mapping ={}
        self.reg = reg
        self._dim = None
        
    def fit(self, X, y, cols_to_encode = None, extra_numeric_cols=None):
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X should be DataFrame type')
        
        if isinstance(y, six.string_types):
            if y in X.columns:
                self._dim = X.shape[1] - 1
                X = X.rename(columns={y:'_y_'})
            else:
                raise ValueError('y not in X.columns during fit')
        else:
            self._dim = X.shape[1]
            y = pd.Series(y, name='_y_')
            X = X.join(y)
        # target label as '_y_'
        X['_y_'] = X['_y_'].astype(int)   
        # get encoder columns 
        if cols_to_encode is None:
            cols = self.get_encode_cols(X)
            cols += list(extra_numeric_cols) if extra_numeric_cols is not None \
                    else []
        else:
            cols = cols_to_encode    
        # re-order cols by original order 
        self.encode_cols = sorted(list(set(cols)), key=X.columns.get_loc) 
        if len(self.encode_cols)==0:
            return self     
            
        data_to_encode = X[self.encode_cols+['_y_']]
        # convert na to a pre-defined value 
        data_to_encode.fillna(self.replace_na, downcast='infer',inplace=True)
        data_to_encode[self.encode_cols] = data_to_encode[self.encode_cols].astype(str)
        
        self._pos = data_to_encode['_y_'].sum()  # global positive count
        self._neg = len(data_to_encode['_y_']) - self._pos # global negative count 
        maps ={}
        for col in self.encode_cols:
            woe = self._compute_woe(data_to_encode, col, '_y_') # return series
            maps[col] = woe.to_dict()  # sereis can not sereisable
            
        self.mapping = maps
        
        return self
            
    def _compute_woe(self, df, var, y='_y_'):
        grp = df[y].groupby(df[var]).agg(['sum',lambda x: x.count()-x.sum()])
        grp = grp.rename(columns={'sum':'pos', '<lambda_0>':'neg'})
        #use bayesian prior value to avoid dividing by zero
        woe = np.log((grp['pos']+self.reg)/(grp['neg']+self.reg)) - \
              np.log((self._pos+2*self.reg)/(self._neg+2*self.reg))             
        if not self.dummy_na and str(self.replace_na) in woe:
            woe[str(self.replace_na)] = 0.0          
        return woe
    
    def transform(self, X, y=None, inplace=False):
        if not isinstance(X, pd.DataFrame):
            raise ValueError('X should be DataFrame type')
        if isinstance(y, six.string_types) and y in X.columns:
            if  self._dim != X.shape[1] -1:
                raise ValueError('dimension error')
        # elif self._dim != X.shape[1]:
            # raise ValueError('dimension error')
        
        if not self.encode_cols:
            return X
        
        if not inplace:
            X = X.copy()
        
        X[self.encode_cols] = X[self.encode_cols].fillna(self.replace_na,
                                     downcast = 'infer').astype(str)
        msg = "unseen category {} in column '{}'"
        for col in self.encode_cols:
            tmp = X[col].map(self.mapping[col]) # unseen value filled with NaN
            #handle unknown value
            if any(tmp.isnull()):
                if self.handle_unknown == 'error':
                    raise ValueError(msg.format(set(X[col][tmp.isnull()].values), col))
                elif self.handle_unknown =='zero':
                    print(msg.format(set(X[col][tmp.isnull()].values), col))
                    tmp = tmp.fillna(0.0)  
            X.loc[:,col] = tmp
        return X     
         
if __name__ == '__main__':

    df1 = pd.DataFrame({'A': ['a', 'b', 'a'],
                        'B': [1,3,2],
                        'C': [1, 2, 3]})
    df2 = pd.DataFrame({'A': ['a', np.nan, 'a'],
                        'B': [2, 2, 3],
                        'D': [1,2,3]})
    

    df = pd.DataFrame([[1,2,3,'a',2.0],[1,5,4,'6',3.0],[2,3,4,5,6],
                       [2.0,3,4,5,np.nan]],columns=['x','y','z','j','k'])
                    
#    df.index=['1','2','3','4']
#    ohe = OneHotEncoder(dummy_na=False)
#    ohe.fit(df1)  #extra_numeric_cols=['y','x','j']
#    ret = ohe.transform(df2)
#    print(ret)
    
#    lbe = LabelEncoder()
#    lbe.fit(df2, extra_numeric_cols=['B'])
#    ret = lbe.transform(df2)
#    print(df2)
#    print(ret)
#    print(df1)
#    ret = lbe.transform(df1)
#    print(ret)
    
    # en = TargetEncoder()
    # en.fit(df1,y=np.array([1,0,1]),extra_numeric_cols=['B'])
    # ret = en.transform(df1)
    # print(ret)
    # ret2 = en.transform(df2)
    y=[1,0,1,1]
    en = WoeEncoder()
    en.fit(df, y=y,extra_numeric_cols=['k'])
    print(df)
    print(y)
    print(en.mapping)
    print(en.transform(df))
    
   