import datetime
import time
import gc
import json
import os
from collections import abc

import numpy as np
import pandas as pd
from sklearn import preprocessing
# from datetime import datetime
# from config import cfg
from sklearn.model_selection import train_test_split

# from pyspark.sql import SparkSession
# from pyspark.ml.feature import VectorAssembler, StandardScaler
# from pyspark.sql import functions as funcs

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


class MixDataProcessor(object):
    def __init__(self, conf, epsilon=1e-8, skew_double=False, colocate_embedding=False, log_transform=False):
        """
        skew_double: if true, discritize skewed numeric feature 
        colocate_embedding: if true all features share a big embedding instance, else each feature \
                  corresponds to a specific embedding instance
        log_transform: use log transform for numerical feature
        """
        self.conf=conf
        self.epsilon=epsilon
        self.drop_cols=[]
        self.double_cols=[]
        self.skew_cols=[]
        self.cat_cols=[]

        self.feature_index_map={}
        self.scaler = {}
        self.skew_double = conf.get('skew_double') or  skew_double 
        self.colocate_embedding = conf.get('colocate_embedding') or colocate_embedding
        self.log_transform = conf.get('log_transform') or  log_transform
        # self.skew_threshold = skew_threshold
        # self.skew_bins

        self.field_dim = 0
        self.feature_dim = 0

    @staticmethod
    def log2map(value):
        if value>4:
            return np.log2(value**2)  # log2(x^2)
        elif value<-4:
            return -np.log2(value**2)
        else:
            return value


    def fit(self, df,  mannual_cat_cols=None, exclude_cols=None):
        """
        mannual_cat_cols: specially treating some numerical cols as categorical type
        exclude_cols: remove some cols that cannot be used to train the model
        """
        columns = df.columns.tolist()
        if isinstance(exclude_cols,str) and exclude_cols in columns:
            columns.remove(exclude_cols)
        elif isinstance(exclude_cols, abc.Iterable):
            exclude_cols = [col for col in exclude_cols if col in columns]
            for col in exclude_cols:
                columns.remove(col)
            # df = df.drop(columns=[y])
            
        # mannual_cat_cols = [] if mannual_cat_cols is None else list(mannual_cat_cols)
        mannual_cat_cols = [] if mannual_cat_cols is None else \
            [col for col in mannual_cat_cols if col in columns]

        cat_cols = df.loc[:,columns].select_dtypes(include=['object','category','datetime']).columns.tolist()
        cat_cols = set(cat_cols + mannual_cat_cols)
        cat_cols = sorted(cat_cols,key=df.columns.get_loc)

        double_cols = df.loc[:,columns].select_dtypes(include=[np.number]).columns.tolist()
        double_cols = [col for col in double_cols if col not in cat_cols]

        # drop null cols
        nulls = df.loc[:,columns].isnull().sum(axis=0)/df.shape[0]
        cat_nulls = nulls[nulls.index.isin(cat_cols)]
        doubel_nulls = nulls[nulls.index.isin(double_cols)]
        drop_cat_cols  = cat_nulls[cat_nulls>self.conf['drop_cat_na_ratio']].index.tolist()
        drop_double_cols = doubel_nulls[doubel_nulls>self.conf['drop_double_na_ratio']].index.tolist()

        cat_cols = [col for col in cat_cols if col not in drop_cat_cols]
        double_cols = [ col for col in double_cols if col not in drop_double_cols]
        
        # treat skewed double feature as categorical type 
        drop_skew_cols = []
        skew_cols = []
        if self.skew_double: 
            skews = df[double_cols].skew(skipna=True)
            self.skew_threshold = self.conf.get('skew_threshold',5)
            self.skew_bins = {}
            skew_cols = skews[abs(skews)>self.skew_threshold].index.tolist()
            bins = self.conf.get('skew_bins',10)
            # print('bins',bins)
            for col in skew_cols:
                col_bins = min(df[col].unique().size, bins)
                _, buckets = pd.qcut(df[col], q=col_bins, labels=None, retbins=True, duplicates='drop')
                if buckets.size<3: # at least 2 buckets 
                    drop_skew_cols.append(col)
                    continue
                self.skew_bins[col] = buckets
                # print('{},len={}'.format(col,len(buckets)))
        
            double_cols = [col for col in double_cols if col not in skew_cols]
            skew_cols = list(self.skew_bins.keys())

        #log_transformer log2(z)^2  if z>2
        if self.log_transform:
            tmp = df[double_cols].astype(float).applymap(self.log2map)
        else:
            tmp = df[double_cols].astype(float)
       
        mean = tmp.mean()
        std =  tmp.std()
        vmin = tmp.min()
        vmax = tmp.max()
         
        # drop double-col with only singletion value
        std_an_cols = std.index[std.isnull()].tolist()
        minmax_cols = vmin[vmin==vmax].index.tolist()  #
        drop_singleton_cols = std_an_cols + minmax_cols
        for col in drop_singleton_cols:
            double_cols.remove(col)

        drop_cols = drop_singleton_cols + drop_cat_cols + drop_double_cols + drop_skew_cols
        columns = [col for col in columns if col not in drop_cols]
    
        # feature value to embedding id
        cnt = 0
        if self.colocate_embedding: # id increase continuously
            for col in columns:
                if col in cat_cols:
                    tmp = df[col].fillna(self.conf['cat_na_sentinel'], downcast='infer').astype(str)
                    us = tmp.unique()
                    self.feature_index_map[col] = dict(zip(us,range(cnt,cnt+len(us))))
                    cnt += len(us)
                elif col in double_cols:
                    self.feature_index_map[col] = cnt 
                    cnt += 1
                elif col in skew_cols:
                    us = list(range(len(self.skew_bins[col]) + 1))
                    us.append(-1) # -1 for na value
                    self.feature_index_map[col] = dict(zip(us,range(cnt,cnt+len(us)))) #auto convert us to str
                    cnt += len(us)
                else:
                    raise ValueError('{} not in target columns'.format(col))
        else:
            for col in columns:
                if col in cat_cols:
                    tmp = df[col].fillna(self.conf['cat_na_sentinel'], downcast='infer').astype(str)
                    us = tmp.unique()
                    self.feature_index_map[col] = dict(zip(us,range(len(us))))
                    cnt += len(us)
                elif col in double_cols:
                    self.feature_index_map[col] = 0 
                    cnt += 1
                elif col in skew_cols:
                    us = list(range(len(self.skew_bins[col]) + 1))
                    us.append(-1) # -1 for na value, when transform need fillna(-1) 
                    self.feature_index_map[col] = dict(zip(us,range(len(us)))) #auto convert us to str
                    cnt += len(us)
                else:
                    raise ValueError('{} not in target columns'.format(col))

        self.cat_cols = cat_cols 
        self.double_cols = double_cols
        self.skew_cols = skew_cols
        self.drop_cols = drop_cols

        self.scaler['mean'] = mean.drop(index=drop_singleton_cols)
        self.scaler['std'] = std.drop(index=drop_singleton_cols)
        # self.scaler['min'] = vmin.drop(index=drop_singleton_cols)
        # self.scaler['max'] = vmax.drop(index=drop_singleton_cols)
        self.field_dim = len(self.feature_index_map)
        self.feature_dim  = cnt

        return self
    
    def transform(self, df, y=None, inplace=False, double_type='normal'):
        """ 
        double_types: options in ('normal','minmax')
        """
        # if isinstance(y,str) and y in df.columns:
        #     target_df = df[y]
        # elif isinstance(y, abc.Iterable):
        #     y = [_ for _ in y if _ in df.columns]
        #     target_df = df[y]
        
        diff_cols = set(self.cat_cols+self.double_cols).difference(df.columns.tolist())
        if len(diff_cols)>0:
            raise ValueError('missing features:{}'.format(diff_cols))
        
        if not inplace:
            df = df.copy()
            gc.collect()
        
        if self.log_transform:
            df[self.double_cols] = df[self.double_cols].astype(float).applymap(self.log2map)

        df[self.double_cols] = df[self.double_cols].fillna(self.scaler['mean'])
        
        if double_type == 'normal':
            df[self.double_cols] = (df[self.double_cols] - self.scaler['mean'])/(self.scaler['std'] + self.epsilon)
        
        #TODO(cut edge)
        elif double_type == 'minmax':
            lbound = 0
            hbould = 1

            df[self.double_cols] = lbound + (df[self.double_cols] - self.scaler['min'])/(self.scaler['max']-self.scaler['min'])*(hbould-lbound)
        
        for col in self.cat_cols:
            df[col] = df[col].fillna(self.conf['cat_na_sentinel'], downcast='infer').astype(str)
            df[col] = df[col].map(self.feature_index_map[col])

        for col in self.skew_cols:
            df[col] = df[col].map(lambda x: np.searchsorted(self.skew_bins[col],x),na_action='ignore')  #
            print('max {}={}'.format(col, df[col].max()))
            df[col] = df[col].fillna(-1).astype(int)
            df[col] = df[col].map(self.feature_index_map[col])

        return df

def convert_model_level_2(x):
    if x.startswith('a'):
        return 'a'
    elif x.startswith('reno'):
        return 'reno'
    elif x.startswith('k'):
        return 'k'
    elif x.startswith('find'):
        return 'find'
    elif x.startswith('r'):
        return 'r'
    else:
        return 'other'

    
if __name__ == '__main__':
    import yaml
    import pickle
    import argparse
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--pred',action='store_true')
    parser.add_argument('--colocate',action='store_true')
    parser.add_argument('--data',type=str,required=False)
    args = parser.parse_args()
    
    args_data = args.data if args.data else None
    TRAIN = not(args.pred)
   
    base_path = os.path.abspath('./')
    cfg = yaml.load(open(os.path.join(base_path,'config.yaml'),'r'))
    
    if TRAIN:
        print('processing: train data')
        sign ='_colocate' if cfg['preprocessing']['colocate_embedding'] or args.colocate else ''
        print('colocate_emb:',sign)
        
        raw_data = args_data if args_data is not None  else cfg.pop('raw_data')
        assert raw_data is not None
        
        tmp_file = {}
        tmp_file['base_path'] = base_path
        tmp_file['raw_data'] = raw_data
        tmp_file['data_pkl']  = os.path.splitext(raw_data)[0]+'{}.pkl'.format(sign)#
        tmp_file['feat_dict'] = os.path.join(base_path, 'tmp/features{}_'.format(sign)+time.strftime('%Y%m%d',time.localtime())+'.dict')
        tmp_file['feat_pkl'] = os.path.join(base_path,'tmp/features{}_'.format(sign)+time.strftime('%Y%m%d',time.localtime())+'.pkl')
        cfg.update({'file':tmp_file})
                                   
        # cfg = cfg.pop('preprocessing')
#         df = pd.read_csv(raw_data, sep='\t', encoding='utf-8', error_bad_lines=False, nrows=None)
#         df.columns = df.columns.map(lambda x: x.split('.')[1])
        df= pd.read_csv(raw_data)
        df = df.rename(columns={'apply_label':'label'})
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
            
        drop_cols = ['user_id', 'send_time', 'group', 'cutoff_dayno','cutoff_month_day','loan_label']
        non_na= df.count()/len(df)
        non_na_threshold=0.005
        na_cols = non_na[non_na<non_na_threshold].index.tolist()
        na_cols
        df = df.drop(columns=na_cols+drop_cols)
        # df_pos = df[df.label==1]
        # df_neg = df[df.label==0].sample(frac=0.5, replace=False)
        # df = pd.concat([df_neg,df_pos],axis=0,ignore_index=True)
        # print(df.label.value_counts())
        # df['model_level_2'] = df['model_level_2'].str.lower().map(convert_model_level_2)
        
        mdp = MixDataProcessor(cfg['preprocessing'])
        mdp.fit(df, mannual_cat_cols=['age','is_be_married'], exclude_cols= ['label','imei','ssoid', 'loan_label'])
        df_out = mdp.transform(df, double_type='normal')
        df_out = df_out.drop(columns=mdp.drop_cols)
        
        df_out.to_pickle(tmp_file['data_pkl'])

        mdp.sparse_feature_columns = mdp.cat_cols
        mdp.sparse_feature_size={k: len(mdp.feature_index_map[k])  for k in mdp.sparse_feature_columns}
        mdp.dense_feature_columns = mdp.double_cols

        json.dump(mdp.__dict__, open(tmp_file['feat_dict'],'w+'), cls=custom_json_serialize, ensure_ascii=False, indent=2)
        pickle.dump(mdp,open(tmp_file['feat_pkl'],'wb+'))
        yaml.dump(cfg, stream=open(os.path.join(base_path,'_config{}.yaml'.format(sign)),'w+'))
     
        
    # NOTE test 
    else:
#         cfg = yaml.load(open('../config.yaml','r'))
        raw_data =  cfg.pop('raw_data')
        cfg = cfg.pop('preprocessing')
        # for key in preprocessing:
        #     cfg[key] = preprocessing[key]
        
        df = pd.read_csv(raw_data, encoding='utf-8', error_bad_lines=False, nrows=None)
        df['model_level_2'] = df['model_level_2'].str.lower().map(convert_model_level_2)

        # st = DataNumerify(cfg, skew_double=False)
        # st.fit(df, mannual_cat_cols=['age'], exclude_cols=['label','imei'])
        st = pickle.load(open('../server/lookalike_autodis_log_level2.pkl','rb'))
        df_out = st.transform(df, double_type='normal')
        print(df_out.shape)
        df_out.isna().sum()
        df_out.to_pickle('../data/lookalike_autodis_log_level2_var.pkl')

    

