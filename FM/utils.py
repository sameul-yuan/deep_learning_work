#utf-8
import six
import json 
import pandas as pd 
import numpy as np
import datetime
from collections import OrderedDict
from sklean.base import BaseEstimator, TransformerMixin

color2num=dict(gray=30, red=31,green=32,yellow=33,blue=34,magenta=35,cyan=36,white=37,crimson=38)
def colorize(string, color, bold=False, highlight=False):
    attr=[]
    num = color2num[color]
    if highlight:
        num +=10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' %(';'.join(attr),string)

class custom_json_serialize(json.JSONEncoder):
    def default(self,obj):
        if isinstance(obj,np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        elif isinstance(obj,datetime.date):
            return obj.strftime('%Y/%m/%d')
        elif isinstance(obj,(pd.DataFrame, pd.Series)):
            return obj.to_dict()
        else:
            return super(custom_json_serialize,self).default(obj)

            
class Dataset(object):
    def __init__(self,xv,xi,y, batch_size=32,shuffle=False):
        self.xv = np.asarray(xv)
        self.xi = np.asarray(xi)
        self.y = np.asarray(y)
        self.batch_size = batch_size
        self.size = self.xv.shape[0]
        self.begin =0
        self.shuffle=shuffle
        self.index = np.arange(self.size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.begin<self.size:
            self.end=min(self.begin+self.batch_size,self.size)
            index = self.index[self.begin:self.end]
            xv = self.xv[index].tolist()
            xi = self.xi[index].tolist()
            y = self.y[index]
            self.begin = self.begin + self.batch_size
            return xv, xi,y
        else:
            self.begin = 0
            if self.shuffle:
                self.index = np.random.permutation(self.size)
            raise StopIteration

class DeepFMData(BaseEstimator,TransformerMixin):
    def __init__(self, drop_na_ratio=0.95, category_threshold=20, skew_thershod=None, replace_cat_na=-99,
                double_process='z-score',**kwargs):
        self._reset()
        self.drop_na_ratio = drop_na_ratio
        self.category_threshold = category_threshold
        self.skew_thershod = skew_thershod
        self.replace_cat_na = replace_cat_na
        self.double_process = double_process
        self.kwargs = kwargs
        assert self.double_process in ('z-score','min-max')
    
    def _reset(self):
        self._field_dim=None
        self._feature_dim=None
        self._double_cols=[]
        self.field_names=[]
        self.feature_index_map={}
        self.cols_to_encode=None
        self.drop_cols=[]
        self.long_tail_cols=[]
        self.scaler ={}

    @staticmethod
    def get_encode_cols(df, dtype_to_encode=['object','category']):
        cols =df.select_dtypes(include=dtype_to_encode).columns.tolist()
        return cols
    
    def fit(self, X, y=None, cols_to_encode=None, extra_numeric_cols=None):
        print('fit'.center(50,'-'))
        assert isinstance(X,pd.DataFrame),' X should be dataframe'
        self._reset()
        self.field_names =X.columns
        extra_numeric_cols = [] if extra_numeric_cols is None else list(extra_numeric_cols)
        if isinstance(y, six.string_types):
            if y in X.columns:
                self.field_names.remove(y)
                X = X.drop(columns=[y])
            else:
                raise ValueError("y not in X.columns during {}.fit".format(self.__class__.__name__))
            
        nulls = X.isnull().sum(axis=0)/len(X)
        drop_null_cols = nulls[nulls>=self.drop_na_ratio].index.tolist()
        X = X.drop(columns=drop_null_cols)
        self.drop_cols.extend(drop_null_cols)
        print('drop_null_cols({})={}'.format(len(drop_null_cols),drop_null_cols))

        if self.skew_threshold is not None:
            print('long_tail_discritize...')
            double_cols = self.get_encode_cols(X, dtype_to_encode=[np.number])
            double_cols = [col for col in double_cols if col not in extra_numeric_cols and col !=y]
            skews = X[double_cols].skew(skipna=True)
            long_tail_cols = skews[abs(skews)>self.skew_threshold].index.tolist()
            drop_tail_cols=[] 
            self.tail_bins={}
            self.tail_mapping={}
            self.buckets = self.kwargs.get('bins',5)
            self.labels = list(map(chr,ord('a')+np.arange(self.buckets)))
            for col in long_tail_cols:
                if X[col].unique().size<self.buckets:
                    drop_tail_cols.append(col)
                    continue
                _, bins = pd.qcut(X[col],q=self.buckets,lables=None,retbins=True, duplicates='drop')
                if bins.size<3:
                    drop_tail_cols.append(col)
                    continue
                self.tail_bins[col] = bins.tolist()
                self.tail_mapping[col] = OrderedDict({chr(ord('a')+i): i for i in range(bins.size-1)})
                if X[col].isna().sum()/len(X)>0.05:
                    self.tail_mapping[col]['null'] = bins.size-1
                self.long_tail_cols.append(col)
            
            print('drop_long_tail_cols({})={}'.format(len(drop_tail_cols),drop_tail_cols))
            X = X.drop(columns=drop_tail_cols)
            self.drop_cols.extend(drop_tail_cols)
            
            if cols_to_encode is None:
                cols_to_encode = self.get_encode_cols(X)
            self.cols_to_encode = list(set(cols_to_encode))
           
            double_cols = [col for col in X.columns if col not in self.cols_to_encode and col not in self.long_tail_cols]
            mean = X[double_cols].mean()
            std = X[double_cols].std()
            vmin = X[double_cols].min()
            vmax = X[double_cols].max()
            std_ann_cols = std.index[std.isnull()].tolist() #avoid featuer with one numeric value and nan
            minmax_cols = vmin[vmin==vmax].index.tolist()
            drop_double_cols = std_ann_cols+minmax_cols
            X=X.drop(columns=drop_double_cols)
            self.drop_cols.extend(drop_double_cols)

            drop_cat_cols=[]
            cnt = 0 
            for col in X.columns:
                if col in self.cols_to_encode:
                    tmp = X[col].fillna(self.replace_cat_na,downcast='infer').astype(str)
                    us =tmp.unique()
                    if us.size>self.category_threshold:
                        drop_cat_cols.append(col)
                    else:
                        self.feature_index_map[col] = dict(zip(us,range(cnt,cngt+len(us))))
                        cnt+=len(us)
                elif col in self.long_tail_cols:
                    us = self.tail_mapping[col].keys()
                    self.feature_index_map[col]=dict(zip(us,range(cnt,cnt+len(us))))
                    cnt+=len(us)
                else:
                    self.feature_index_map[col]=cnt 
                    cnt +=1 
                    self._double_cols.append(col)
                
            self.drop_cols.extend(drop_cat_cols)
            self.cols_to_encode = [col for col in self.cols_to_encode if col not in drop_cat_cols]

            self.scaler['mean'] = mean.drop(index=drop_double_cols)
            self.scaler['std']= std.drop(index=drop_double_cols)
            if self.double_process =='min-max':
                self.scaler['min'] = vmin.drop(index=drop_double_cols)
                self.scaler['max'] = vmax.drop(index=drop_double_cols)

            self._field_dim = len(self.feature_index_map)
            self._feature_dim = cnt 
        return self

    def transform(self, X, y=None, inplace=False, normalize_double=True):
        print('transform'.center(50,'-'))
        assert isinstance(X,pd.DataFrame)
        columns = X.columns.tolist()
        if y is not None and y in X.columns:
            columns.remove(y)
            X=X.drop(columns=[y])
        diffs = set(columns).symmetric_difference(self.field_names)
        if diffs:
            raise ValueError('feature names mismatch during {}.fit'.format(self.__class__.__name__))
        Xc = X.drop(columns=self.drop_cols)
        del X 

        Xc[self.cols_to_encode] = Xc[self.cols_to_encode].fillna(self.replace_cat_na,downcast='infer').astype(str)

        for col in self.long_tail_cols:
            buckets = len(self.tail_bins[col])-1
            idx = list(range(buckets+2))
            val =[self.labels[0],*self.labels[:buckets],self.labels[buckets-1]]
            idx2val =dict(zip(idx,val))
            Xc[col] = Xc[col].map(lambda x:np.searchsorted(self.tail_bins[col],x),na_action='ignore')
            Xc[col] = Xc[col].map(idx2val)
            if 'null'  in self.tail_mapping[col]:
                Xc[col] = Xc[col].fillna('null')
            
        if normalize_double:
            Xc[self._double_cols] = Xc[self._double_cols].fillna(self.scaler['mean'])
            if self.double_process =='z-score':
                Xc[self._double_cols] = (Xc[self._double_cols] - self.scaler['mean'])/(self.scaler['std']+1e-8)
            else:
                lb = self.kwargs.get('lbound',0)
                hb = self.kwargs.get('hbound',1)
                Xc[self._double_cols] = lb +(Xc[self._double_cols] - self.scaler['min'])/(self.scaler['max']-self.scaler['min'])*(hb-lb)
                Xc[self._double_cols] = Xc[self._double_cols].applymap(lambda x: min(max(x,lb),hb))
        Xi = Xc.copy()
        cat_cols = list(sorted(self.cols_to_encode+self.long_tail_cols,key=Xc.columns.get_loc))
        for col in Xc.columns:
            if col in cat_cols:
                tmp = Xi[col].map(self.feature_index_map[col])
                if tmp.isnull().sum()>0:
                    print('unseen category{} in feature {}'.format(set(Xi[col].unique()).difference(set(self.feature_index_map[col])),col))
                    vmin = min(self.feature_index_map[col].values())
                    null_index =tmp[tmp.isnull()].index
                    tmp = tmp.fillna(vmin)
                else:
                    null_index =[] 
                Xi[col]=tmp 
                Xc[col] =1.0 
                Xc[col][null_index]=0 
            else：
                xi[col] = self.feature_index_map[col]
            Xi[col]= Xi[col].astype(int)
        Xi = Xi.values.tolist()
        Xv = Xc.values.tolist()

        return Xv, Xi



        


               


