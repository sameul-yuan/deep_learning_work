import datetime
import os
import pandas as pd
import numpy as np
import datetime
from collections import Counter

import os
import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import pandas as pd 
from sklearn.metrics import roc_auc_score
from torch.utils.data import sampler, DataLoader, Dataset

from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, FloatType, StringType, DoubleType, IntegerType
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, PandasUDFType,row_number, max, broadcast,udf, desc, asc
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .config("spark.files.overwrite", "true") \
    .getOrCreate()
sc = spark.sparkContext

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
spark.conf.set('spark.sql.adaptive.enabled', 'false')
# spark.conf.set('spark.sql.shuffle.partitions', '2000')


class DateUtils():
    @staticmethod
    def input_format(date_str):
        import re
        if re.match('^2\d{7}$', date_str):
            return datetime.date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:]))
        elif re.match('^\d{4}-\d{2}-\d{2}$', date_str):
            return datetime.date(int(date_str[:4]), int(date_str[5:7]), int(date_str[8:]))
        else:
            raise ValueError(f"input date format must be YYYYMMDD or YYYY-MM-DD,but got {date_str}")
    
    @staticmethod
    def ndays_ago(date=None, delta_dyas=90):
        date = DateUtils.input_format(date)
        delta = datetime.timedelta(days=delta_dyas)
        begin_dayno = date - delta
        #end_dayno = date
        return begin_dayno.strftime("%Y%m%d")
    
    @staticmethod
    def gen_retrace_dates(end_date=None, delta_dyas=1):
        end_date = str(end_date)
        end = DateUtils.input_format(end_date)
        delta = datetime.timedelta(days=1)
        date_list = []
        d = end
        for i in range(1, delta_dyas + 1):
            d -= delta
            date_list.append(d.strftime("%Y%m%d"))
        return date_list
    
    @staticmethod
    def gen_dates(start_date=None,end_date=None):
        end_date = str(end_date)
        start_date = str(start_date)
        end = DateUtils.input_format(end_date)
        start = DateUtils.input_format(start_date)
        delta = datetime.timedelta(days=1)
        date_list = []
        d = end
        date_list.append(d.strftime("%Y%m%d"))
        while 1:
            d -= delta
            date_list.append(d.strftime("%Y%m%d"))
            if d==start:
                break
        return date_list[: :-1]
    
    @staticmethod
    def date_sub(start_date=None,end_date=None):
        end_date = str(end_date)
        start_date = str(start_date)
        end = DateUtils.input_format(end_date)
        start = DateUtils.input_format(start_date)
        delta = end-start
        return delta.days


def activation_layer(act_name):
    """Construct activation layers

    Args:
        act_name: str or nn.Module, name of activation function
    Return:
        act_layer: activation layer
    """
    if isinstance(act_name, str):
        if act_name.lower() == 'sigmoid':
            act_layer = nn.Sigmoid()
        elif act_name.lower() == 'linear':
            act_layer = Identity()
        elif act_name.lower() == 'relu':
            act_layer = nn.ReLU(inplace=True)
        elif act_name.lower() == 'prelu':
            act_layer = nn.PReLU()
    elif issubclass(act_name, nn.Module):
        act_layer = act_name()
    else:
        raise NotImplementedError

    return act_layer

class DeepFM(torch.nn.Module):
    def __init__(self, onehot_size, field_size, n_dim, hidden_size=[256, 128],padding=False, act_fn=torch.nn.functional.relu):
        super(DeepFM, self).__init__()
        self.onehot_size = onehot_size  # 所有特征的分bin之后的bin的个数 174353
        self.field_size  = field_size # 特征数 65226
        self.n_dim  = n_dim #
        self.act_fn = act_fn
        self.loss = []
        self.hidden_layers = []
        self.bn_layers = []
        #self.dropout = torch.nn.Dropout(0.2) #jww
        
        n_input = field_size * n_dim
        # 判断是否为str类型
        if isinstance(hidden_size, str): 
            hidden_size = map(int, hidden_size.split(","))
        for i, h in enumerate(hidden_size):
            layer = torch.nn.Linear(n_input, h)
            torch.nn.init.xavier_uniform_(layer.weight.data)
            bn = torch.nn.BatchNorm1d(h)
            self.bn_layers.append(bn)
            self.hidden_layers.append(layer)
            self.add_module(f"hidden_{i}", layer)
            self.add_module(f"bn_{i}", bn)
            n_input = h
        self.dnn_output = torch.nn.Linear(n_input, 1)
        self.embedding_layer = torch.nn.Embedding(onehot_size, n_dim)
        self.w = torch.nn.Embedding(onehot_size, 1) # 1维，每个特征 i 对应一个权重 w(i) torch.nn.Embedding(m, n) : m 表示单词的总数目，n 表示词嵌入的维度

        # 初始化要注意，这两个embedding层其实是当线性层用的，因此不可使用embedding层默认的初始化而应该使用Linear层的初始化，否则无法收敛
        torch.nn.init.normal_( self.embedding_layer.weight, 0.0, np.sqrt(2/field_size/(field_size-1)/n_dim) )
        torch.nn.init.uniform_( self.w.weight, -np.sqrt(1/onehot_size), np.sqrt(1/onehot_size) )

    def forward(self, x, v=None):
        # x: batch_size, field_size(特征 multihot后的下标)
        # v: (batch_size, field_size, 1) or None(若有连续特征，则为其具体的特征值)
        batch_size = x.shape[0]

        # embedding部分
        emb = self.embedding_layer(x) # batch_size, field_size, n_dim
        xv = emb if v is None else emb * v

        # deep部分
        h = torch.flatten(xv, 1) # batch_size, field_size*n_dim
        for layer, bn in zip(self.hidden_layers, self.bn_layers):
            h = bn(self.act_fn(layer(h)))
        y_dnn = self.dnn_output(h)

        # fm部分
        # 一阶部分，y1 = w1*x1 + w2*x2 + ...
        wx = self.w(x) if v is None else self.w(x) * v
        y1 = wx.sum(dim=1)
        # 二阶部分
        x_sum = xv.sum(dim=1) # shape=(batch_size, n_dim)，即x1*v1+x2*v2+...
        x1 = (x_sum * x_sum).sum(dim=1, keepdim=True) # shape=(batch_size, 1)，即(x1*v1+x2*v2+...)^2
        x2 = (xv * xv).sum(dim=[1, 2]).unsqueeze(-1) # shape=(batch_size, 1)，即(x1*v1)^2+(x2*v2)^2+...
        y2 = 0.5 * (x1 - x2)
        
        return  torch.sigmoid(y1 + y_dnn + y2)
    
def transform_field_id_fn(feat_str):
    x = np.copy(raw_fields)
    fields = np.array([kv.split(':') for kv in feat_str.split(' ')], dtype=np.long)
    x[fields[:, 0], 1] = fields[:, 1]
    feat = x[:,1].tolist()
    return feat

def read_mapping_file(file_path):
    with open(file_path) as f:
        line = f.readline()
        fields = np.array([kv.split(':') for kv in line.split(' ')], dtype=np.long)
        return fields

def batch_predict(BC_MODEL, field_size=None, onehot_size=None):
    @pandas_udf('float')
    def predict_func(feat_col: pd.Series) -> pd.Series:
        feats = feat_col.apply(lambda feat_str:transform_field_id_fn(feat_str))
        feats = torch.LongTensor(feats)
        BC_MODEL.eval()
        with torch.no_grad():
            pred = BC_MODEL(feats)
            #pred = torch.sigmoid(pred)
        return pd.Series(pred.squeeze(1).tolist())
    return predict_func

def score_main(model_path,map_path, feat_table, score_table, dayno):
    
    field_size=30861
    onehot_size=109579
    model= DeepFM(onehot_size=onehot_size, field_size=field_size, hidden_size=[128, 64], n_dim=32)
    
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    BC_MODEL = sc.broadcast(model).value

    # 特征表
    feat_df = spark.sql(f'select * from {feat_table} where dayno={dayno}').repartition(3000).persist()
    #feat_df = spark.sql("select * from loan_tmp.jww_deepfm_test")
    feat_df.show(3)

    # 预测
    predict_udf = batch_predict(BC_MODEL, field_size=field_size, onehot_size=onehot_size)
    score_df = feat_df.select('imei', predict_udf('field_feat').alias('score'))
    return score_df

if __name__ == '__main__':
    import sys
    feat_table = sys.argv[1]
    score_table = sys.argv[2]
    dayno = sys.argv[3]
    model_path="f_evt_insurance_model_deepfm_click_v1.pt"
    map_path="ins_merge_field2onehot_app.txt"
    raw_fields=read_mapping_file(map_path)
    raw_fields=sc.broadcast(raw_fields).value
    df = score_main(model_path,map_path, feat_table, score_table, dayno)    
    df.registerTempTable('tmp_table')
    spark.sql(f"INSERT OVERWRITE TABLE {score_table} PARTITION(dayno={dayno}) select imei,score from tmp_table")
