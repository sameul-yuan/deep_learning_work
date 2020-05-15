import numpy as np
import os
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.sparse import csr_matrix
import six
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import cycle
# from scipy.stats import pearsonr
# from scipy.stats import boxcox, boxcox_normmax
from collections import OrderedDict
from logger import colorize
import warnings
import gc
warnings.filterwarnings(action='ignore')


class StandardNumerify(BaseEstimator, TransformerMixin):
    """
    preprocessing tabular data so as to  feeding into neural network.

    one-hot for string-type features,  standard normalization for gaussian-like numeric feature, long tail distributions
    are tackled either with transformation( log for right skew, and max-cop for left skew) or discretizing into buckets
    (equal frequency) followed by one-hot.
    y(\lambda) = (y^(\lambda) -1)/\lambda, if \lambda != 0 (left skew)
            or = log(y+1) if \lambda = 0 (right skew)

    parameter
    ---------
    dummy_na:           bool, defualt True,  treat na as one category
    drop_na_ratio:      float, determine the max null ratio of feature that otherwise will be discarded
    unknown_category:   str, 'error' or 'ignore'
    category_threshold: columns with  categories more then this will be discarded, one-hot method used for the others.
    na_sentinel:        value for fillna
    double_preproc:     str, options ('normal', 'minmax'), proprocess for dense numeric  feature
    long_tail_preproc:  str, options('boxcox', 'discretize',None), preprocess for long-tail distribution numeric feature
    kwargs:             dict, supplementary params for previous arguments.
                        ('lbould', 'hbould') for double_preproc='minmax', ('buckets') for long_tail_preproc='discretize'
                        ('skew_threshold') when long_tail_preproc is not None
    examples
    --------
    '>>> data = np.array([['a',1, 2],['b',1,2]])
    '>>> df = pd.DataFrame(data)
    '>>> snf = StandardNumerify()
    '>>> snf.fit(data)
    '>>> ret = snf.transform(data)
    """

    def __init__(self, dummy_na=True, drop_na_ratio=0.8, unknown_category='ignore', category_threshold=30, na_sentinel=-99,
                 double_preproc='normal', long_tail_preproc=None, epsilon= 1e-8, **kwargs):

        self.dummy_na = dummy_na
        self.drop_na_ratio = drop_na_ratio
        self.unknown_category = unknown_category
        self.category_threshold = category_threshold
        self.na_sentinel = na_sentinel
        self.double_preproc = double_preproc
        self.long_tail_preproc = long_tail_preproc
        self.encode_cols = []
        self.double_cols = []
        self.drop_cols = []
        self.long_tail_cols =[]
        self.scaler ={}
        self.mapping = {}
        self._dim = None
        self.epsilon = epsilon
        self._check_kwargs(**kwargs)
        self.kwargs = kwargs

    @staticmethod
    def _check_kwargs(**kwargs):
        args =('lbound','hbound','buckets', 'skew_threshold')
        diffs = set(kwargs.keys()) - set(args)
        if diffs:
            raise  ValueError('unsupported kwargs:{}'.format(diffs))

    @staticmethod
    def get_encode_cols(df, dtype_to_encode=('object', 'category')):
        cols = df.select_dtypes(include=dtype_to_encode).columns.tolist()
        return cols

    def dump(self,file):
        with open(file,'wb+') as fp:
            pickle.dump(self, fp)

    def fit(self, X, y=None, cols_to_encode=None, extra_numeric_cols=None):
        """
        parameter
        ----------
        X: DataFrame  to generate one-hot-encoder rule
        y: label column in DataFrame X if provided

        cols_to_encoders: specify the columns  to be  encoded
        extra_numeric_cols: if cols_to_encoder is provided this param will
           not be used, otherwise all object columns and extra_numeric_cols
           will be encoded.
        """
        print('fitting....')
        assert isinstance(X, DataFrame), 'X should be DataFrame object'
        columns = X.columns.tolist()
        if y is not None:
            if y not in columns:
                raise ValueError('y is not in X.columns during {}.fit'.format(self.__class__.__name__))
            else:
                columns.remove(y)

        self._dim = len(columns)

        # drop null cols
        nulls = X.isnull().sum(axis=0) / len(X)
        drop_null_cols = nulls[nulls>=self.drop_na_ratio].index.tolist()
        X = X.drop(columns=drop_null_cols)
        self.drop_cols.extend(drop_null_cols)
        print(colorize('drop_null_cols({})={}'.format(len(drop_null_cols), drop_null_cols),'blue',True))

        # get encoder columns
        if cols_to_encode is None:
            cols = self.get_encode_cols(X)
            cols += list(extra_numeric_cols) if extra_numeric_cols else []
        else:
            cols = cols_to_encode
        if y in cols:
            cols.remove(y)
        cols = sorted(list(set(cols)), key= columns.index)

        # convert na to sentinel value
        df = X[cols].fillna(self.na_sentinel, downcast='infer').astype(str)

        # generate rules
        colvals={}
        drop_cat_cols =[]
        for col in cols:
            values = df[col].unique().tolist()
            if str(self.na_sentinel) in  values and not self.dummy_na:
                values.remove(str(self.na_sentinel))
            if 1<len(values)<=self.category_threshold:
                colvals[col] = values
            else:
                drop_cat_cols.append(col)
        print(colorize('drop_cat_cols({})={}'.format(len(drop_cat_cols), drop_cat_cols), 'blue', True))
        self.drop_cols.extend(drop_cat_cols)
        self.encode_cols = list(sorted(colvals.keys(), key=df.columns.get_loc))
        self.double_cols = [col for col in columns if col not in self.encode_cols and col not in self.drop_cols]

        for col in self.encode_cols:
            vals = colvals[col]
            self.mapping[col] = OrderedDict({val: i for i, val in enumerate(vals)})
        # cats = df.apply(lambda x: x.unique().__len__(), axis=0)
        # subs = 0 if self.dummy_na else df.apply(lambda x: str(self.na_sentinel) in x.values)
        # cats -= subs
        # self.drop_cols = cats[(cats>=self.category_threshold)|(cats<=1)].index.tolist()
        # self.encode_cols = cats[~cats.index.isin(self.drop_cols)].index.tolist()  # cats.index.difference(drop_cols), turns changed
        # self.double_cols = [col for col in columns if col not in self.encode_cols and col not in self.drop_cols]

        # long-tail-distribution(in double cols)
        if self.long_tail_preproc is not None:
            skews = X[self.double_cols].skew(skipna=True)
            thres = self.kwargs.get('skew_threshold',5)
            lt_cols = skews[abs(skews)>thres].index.tolist()
            print(colorize('long-tail-cols({})={}'.format(len(lt_cols), lt_cols),'blue',True))

            if self.long_tail_preproc == 'discretize':
                drop_tail_cols =[]
                self.tail_bins ={}
                self.tail_mapping ={}
                self.long_tail_cols =[]
                self.buckets = self.kwargs.get('buckets',5)
                self.labels = list(map(chr, ord('a') + np.arange(self.buckets)))
                for col in lt_cols:
                    if X[col].unique().size < self.buckets:
                        drop_tail_cols.append(col)
                        continue
                    _, bins = pd.qcut(X[col], q=self.buckets, labels=None, retbins=True, duplicates='drop')
                    if bins.size<3:  # at least 2 bins
                        drop_tail_cols.append(col)
                        continue
                    self.tail_bins[col] = bins.tolist()
                    self.tail_mapping[col] = OrderedDict({chr(ord('a')+i): i for i in range(bins.size-1)})
                    if (X[col].isna().sum()/len(X))>0.01:
                        self.tail_mapping[col]['null'] = bins.size-1
                    self.double_cols.remove(col)
                    self.long_tail_cols.append(col)

                self.drop_cols.extend(drop_tail_cols)
                self.encode_cols.extend(self.long_tail_cols)

                for col in drop_tail_cols:
                    self.double_cols.remove(col)
                print(colorize('drop_tail_cols({})={}'.format(len(drop_tail_cols),drop_tail_cols),'blue',True))
            else:
                # TODO(yuanyuqing163): implement boxcox transformation
                raise ValueError("boxcox transformation hasn't implemented yet")

        self.scaler['mean'] = X[self.double_cols].mean()
        self.scaler['std'] = X[self.double_cols].std()
        if 'minmax' in self.double_preproc:
            self.scaler['min'] = X[self.double_cols].min()
            self.scaler['max'] = X[self.double_cols].max()
        elif 'normal' not in self.double_preproc:
            raise ValueError('double_process_type = {} not supported yet'.format(self.double_preproc))

        return self

    def transform(self, X, y=None, dtype=None, inplace=False):
        """
        parameter
        -----------
        dtype: specifies the dtype of encoded value
        """
        print('transform....')
        assert isinstance(X, DataFrame), 'X shoule be DataFrame object'
        columns = X.columns.tolist()
        target_df = []
        if y is not None:
            if y not in columns:
                raise ValueError("'y label {}' not in X".format(y))
            else:
                columns.remove(y)
                target_df = [X.loc[:,[y]]]

        assert self._dim == len(columns)

        diff_cols = set(self.encode_cols+self.double_cols).difference(columns)
        if len(diff_cols) > 0:
            raise ValueError("X not includes encoded columns '{}'".format(diff_cols))

        if not inplace:
            X = X.copy()  # X=X.copy(deep=True)
            gc.collect()

        X.drop(self.drop_cols, axis=1, inplace=True)

        X[self.double_cols] = X[self.double_cols].fillna(self.scaler['mean'])
        if 'normal' in self.double_preproc:
            X[self.double_cols] = (X[self.double_cols] - self.scaler['mean']) / (self.scaler['std'] + self.epsilon)

        #TODO：truncate interval [min,max]
        if 'minmax' in self.double_preproc:
            lbound = self.kwargs.get('lbound', 0)
            hbound = self.kwargs.get('hbould',1)
            X[self.double_cols] = lbound + (X[self.double_cols] - self.scaler['min'])/(
                self.scaler['max']-self.scaler['min'])*(hbound-lbound)

        # long_tail_feature
        if self.long_tail_preproc=='discretize':
            print('long_tail_discretize.....')
            for col in self.long_tail_cols:
                buckets = len(self.tail_bins[col])-1
                idx = list(range(buckets + 2))
                val = [self.labels[0], *self.labels[:buckets], self.labels[buckets-1]]
                idx2val = dict(zip(idx, val))
                X[col] = X[col].map(lambda x: np.searchsorted(self.tail_bins[col], x), na_action='ignore')
                X[col] = X[col].map(idx2val)
                if 'null' in self.tail_mapping[col]:
                    X[col] = X[col].fillna('null')
                else:
                    print(colorize("'null' not long tail in '{}'=#{}".format(col, X[col].isna().sum()),'cyan',True))

        elif self.long_tail_preproc == 'boxcox':
            raise ValueError('unsupported long_tail_preproc type'.format(self.long_tail_preproc))

        data_to_encode = X[self.encode_cols].fillna(self.na_sentinel, downcast='infer').astype(str)
        with_dummies = [X[self.double_cols]]
        prefix = self.encode_cols
        prefix_sep = cycle(['_'])
        print('encoding....')
        for (col, pre, sep) in zip(data_to_encode.iteritems(), prefix,
                                   prefix_sep):
            # col is (col_name, col_series) type
            dummy = self._encoder_column(col[1], pre, sep, dtype=dtype)
            with_dummies.append(dummy)

        result = pd.concat(with_dummies+target_df, axis=1)

        return result

    def _encoder_column(self, data, prefix, prefix_sep, dtype):
        if dtype is None:
            dtype = np.uint8

        maps = self.mapping.get(prefix,{}) or self.tail_mapping.get(prefix,{})
        dummy_strs = cycle([u'{prefix}{sep}{val}'])
        dummy_cols = [dummy_str.format(prefix=prefix, sep=prefix_sep, val=str(v))
                      for dummy_str, v in zip(dummy_strs, maps.keys())]

        out_shape =(len(data),len(dummy_cols))

        if isinstance(data, Series):
            index = data.index
        else:
            index = None
        data.reset_index(drop=True,inplace=True)  # data :Series

        data2 = data.map(maps)
        null = data2[data2.isna()].index
        data2 = data2[data2.notna()]
        if not null.empty:
            print(colorize("{} only exist in test data column '{}'".format(set(data[null].values), prefix),'cyan',True))
        row_idxs = data2.index.tolist()
        col_idxs = data2.values.tolist()
        sarr = csr_matrix((np.ones(len(row_idxs)), (row_idxs, col_idxs)), shape=out_shape, dtype=dtype)
        if pd.__version__ >='0.25':
            out = pd.DataFrame.sparse.from_spmatrix(sarr,index=index, columns=dummy_cols) #sparse accessor, out.sparse.to_dense()
            # dense.astype('Sparse[int]'), dense.astype(pd.SparseDtype(int,fill_value=0))
            # out.astype(pd.SparseDtype(int, fill_value=0))
        else:
            out = pd.SparseDataFrame(sarr, index=index, columns=dummy_cols,
                                     default_fill_value=0, dtype=dtype)
        return out.astype(dtype)  # care of row and columns not covered by sarr

if __name__ == '__main__':
    import cProfile
    import pickle
    import sys
    # df1 = pd.DataFrame({'A': ['a', 'b', 'c'],
    #                     'B': [1, np.nan, np.nan],
    #                     'C': [1, 2, 3]})
    # df2 = pd.DataFrame({'A': ['a', 'b', 'a'],
    #                     'B': [2, 2, 3],
    #                     'C': [1, 2, 3]})
    # feature = pd.read_csv("/home/yuanyuqing163/hb_hnb_1125/conf/xgb_importance_gain_hnb.csv")
    # columns = feature['feature'].tolist()
    # columns +=['is_y2']
    print(sys.version_info)
    train_file = "/home/yuanyuqing163/hb_rl/data/raw/train_hnb_150.pkl"
    val_file = "/home/yuanyuqing163/hb_rl/data/raw/val_hnb_150.pkl"
    test_file = "/home/yuanyuqing163/hb_rl/data/raw/test_hnb_150.pkl"
    encoder_file = "/home/yuanyuqing163/hb_hnb_1219/conf/int_need_category_encoder.csv"
    train = pd.read_pickle(train_file)
    print(train.head())
    # print(train.dtypes.tolist())

    extra_cols = pd.read_csv(encoder_file)
    extra_cols = extra_cols['feature'].tolist()
    extra_cols = [col for col in extra_cols if col in train.columns]

    enc = StandardNumerify(category_threshold=20, long_tail_preproc='discretize',drop_na_ratio=0.8)
    enc.fit(train, y='is_y2',extra_numeric_cols=extra_cols)
    train = enc.transform(train, y='is_y2')
    # train = train.sparse.to_dense()
    print(train.head())
    print(train.dtypes)
    print(type(train))
    # print(sum(train.isna()))
    reduce_mem_usage(train)
    print(train.shape)
    print(train.head())
    print(train.dtypes)
    print(type(train))
    print(train.isna().head())
    train.to_pickle('../data/clean/train_hnb_150_tail.pkl')
    print(train.isna().sum().sum())

    val = pd.read_pickle(val_file)
    val = enc.transform(val,y='is_y2')
    # print(val.head())
    # reduce_mem_usage(val)
    print(val.shape)
    print(val.head())
    print(val.isna().sum().sum(0))
    val.to_pickle('../data/clean/val_hnb_150_tail.pkl')

    test = pd.read_pickle(test_file)
    test = enc.transform(test,y='is_y2')
    # print(test.head())
    # reduce_mem_usage(test)
    print(test.shape)
    print(test.head())
    print(test.isna().sum().sum())
    test.to_pickle('../data/clean/test_hnb_150_tail.pkl')


    with open('../model/standardnumpify_150.pkl','wb+') as fp:
        pickle.dump(enc, fp)
   
    # enc = StandardNumerify(category_threshold=20)
    # enc.fit(train,y='is_y2',extra_numeric_cols=extra_cols)
    # pickle.dump(enc, open('../model/standardnumpify.pkl','wb+'))
    # print(enc.__dict__)
    # train = enc.transform(train,  y='is_y2')
    # print(train.shape)
    # print(train.head())
    # train.to_csv('../data/clean/train_hnb.csv',index=False)

    # val = pd.read_pickle(val_file)
    # val = val[columns]
    # val = enc.transform(val,  y='is_y2')
    # val.to_csv('../data/clean/val_hnb.csv',index=False)
    # print(val.shape,val.head())
    
    # with open('../model/standardnumpify_200.pkl','rb') as fp:
    #     enc = pickle.load(fp)



