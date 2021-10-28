
import os
import argparse
import pandas as pd

os.environ['ARROW_PRE_0_15_IPC_FORMAT'] = '1'


# start spark session
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark import SparkFiles

# import spark in-built functions
import pyspark.sql.functions as f
from pyspark.sql.functions import col,when
from pyspark.sql.functions import col, pandas_udf, udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType

# import pandas as pd
import numpy as np
import torch 
from torch.utils.data import dataloader,SequentialSampler
import yaml
import json


from autoint_pyspark import AutoInt,  get_linear_schedule_with_warmup,  PretrainedConfig


parser = argparse.ArgumentParser(description='pytorch_spark_inference')
parser.add_argument('--model_state',type=str,required=True)
parser.add_argument('--config',type=str,required=True)
parser.add_argument('--features',type=str,required=True)
parser.add_argument('--pred_table',type=str, required=True)
parser.add_argument('--result_table',type=str, required=True)
args = parser.parse_args()

feature_cfg = yaml.load(open(args.features,'r'))
all_cfg = yaml.load(open(args.config,'r'))
network_cfg = all_cfg['FM']
    
   
spark = SparkSession.builder \
      .appName("mkt_user_package") \
      .config('spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT',1) \
      .config('spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT',1) \
      .enableHiveSupport() \
      .getOrCreate()

sc = spark.sparkContext

sc.setLogLevel("WARN")

bc_model_state = None
eval_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_predict_model(args):
    feature_cfg = yaml.load(open(args.features,'r'))
    all_cfg = yaml.load(open(args.config,'r'))
    network_cfg = all_cfg['FM']
    train_cfg = all_cfg['train']

    pretrained_cfg  = PretrainedConfig(**all_cfg['self_attention']) # for AutoInt
    pretrained_cfg.initializer_range = network_cfg['init_norm_var']
    pretrained_cfg.hidden_size = network_cfg['embedding_size']

    model = AutoInt(pretrained_cfg, network_cfg, field_size=feature_cfg['field_dim'], feature_size=feature_cfg['feature_dim'])
    # model.load_state_dict(torch.load(args.model_state, map_location=torch.device('cpu')))  #map_Location=torch.device('cpu')
    model.load_state_dict(bc_model_state.value)
    return model

    
#NOTE (udf must return naive python data type)
@udf
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

@udf('float')
def log2map(value):
    if value is None:
        return float('nan')
    if value>2:
        return (np.log2(value)**2).item()  # 
    elif value<-2:
        return (-np.log2(-value)**2).item()
    else:
        return value


@udf('long')
def age_xi(x):
    maps = feature_cfg['feature_index_map']['age']
    return maps.get(x, maps.get('-999'))

@udf('long')
def gender_xi(x):
   maps = feature_cfg['feature_index_map']['gender']
   return maps.get(x, maps.get('-999'))


@udf('long')
def model_level_2_xi(x):
   maps = feature_cfg['feature_index_map']['model_level_2']
   return maps.get(x, maps.get('-999'))

@udf('long')
def city_level_xi(x):
    maps = feature_cfg['feature_index_map']['city_level']
    return maps.get(x, maps.get('-999'))


def preprocess(df, feature_cfg):
    double_cols = feature_cfg['double_cols']
    # feature_index_map = feature_cfg['feature_index_map']
    cat_cols = feature_cfg['cat_cols']
    
    na_maps = {'age':-999}
    for feat in cat_cols:
        na_maps.update({feat:'-999'})
        
    df = df.fillna(na_maps)
    
#     df = df.fillna({'age': -999, 'model_level_2':"-999", 'gender':"-999",'city_level':'-999'})
    
    if 'model_level_2' in df.columns:
        df = df.withColumn('model_level_2', f.lower(df.model_level_2))  # overwrite col
        df = df.withColumn('model_level_2', convert_model_level_2(df.model_level_2))
        
    if 'age' in df.columns:
        df = df.withColumn('age', df.age.cast('string'))
    
    # log2map 
    for col in double_cols:
        df = df.withColumn(col, df[col].cast('float'))
        df = df.withColumn(col, log2map(df[col]))
    
    # normalization
    double_means = feature_cfg['scaler']['mean']
    df = df.fillna(double_means)
    for col in double_cols:
        mean = feature_cfg['scaler']['mean'][col]
        std = feature_cfg['scaler']['std'][col]
        df = df.withColumn(col, (df[col]-mean)/(std+1.0e-8))

    return df


def get_model_input(df, feature_cfg):
    double_cols = feature_cfg['double_cols']
    cat_cols = feature_cfg['cat_cols']
    feature_index_map = feature_cfg['feature_index_map']
    
    cat_xi = []
    cat_xv = []
    for feat in cat_cols:
        if feat == 'model_level_2':
            df = df.withColumn(feat+'_xi', model_level_2_xi(col(feat)))
            df = df.withColumn(feat+'_xv', f.lit(1.0))
        
        elif feat == 'age':
            df = df.withColumn(feat+'_xi',age_xi(col(feat)))
            df = df.withColumn(feat+'_xv', f.lit(1.0))
            
        elif feat =='gender':
            df = df.withColumn(feat+'_xi', gender_xi(col(feat)))
            df = df.withColumn(feat+'_xv', f.lit(1.0))
            
        elif feat =='city_level':
            df = df.withColumn(feat+'_xi',city_level_xi(col(feat)))
            df = df.withColumn(feat+'_xv',f.lit(1.0))
                               
        else:
            raise ValueError('production dataset missing feature {}'.format(feat))
                               
        cat_xi.append(feat+'_xi')
        cat_xv.append(feat+'_xv')

       

    df = df.withColumn('feats_xi', f.array(*(cat_xi + [f.lit(feature_index_map[col]) for col in double_cols]))).drop(*cat_cols)
    df = df.withColumn('feats_xv', f.array(*(cat_xv + double_cols))).drop(*double_cols)
    df = df.drop(*(cat_xi+cat_xv))

#     df = df.withColumn('feats_xi', f.array(*(['model_level_2_xi','age_xi','gender_xi','city_level_xi'] +[f.lit(feature_index_map[col]) for col in double_cols] )))
#     df = df.withColumn('feats_xv', f.array(*([f.lit(1.0) for _ in range(len(cat_cols)) ] + double_cols)))

    df = df.withColumn('feats',f.array('feats_xi','feats_xv'))

    return df


class SerDataset(torch.utils.data.Dataset):
    def __init__(self, ser):
        self.ser = ser
    def __len__(self):
        return len(self.ser)
    def __getitem__(self,i):
        ser = self.ser.iloc[i]
        xi = ser[0]
        xv = ser[1]
        return torch.tensor(xi,dtype=torch.long),torch.tensor(xv,dtype=torch.float32)


@pandas_udf('float')  # returnType: the element type of the return pd.Series
def model_udf_inference(input : pd.Series)->pd.Series:
    dataset = SerDataset(input)
    sampler = SequentialSampler(dataset)
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=1000, num_workers=2)
    # loader = torch.utils.data.DataLoader(input, batch_size=4)

    all_predictions=[]
    eval_model.eval()
    with torch.no_grad():
        for batch in loader:
            # arry = np.asarray(batch)
            # xi  = torch.tensor(arry[:,0],dtype=torch.long)
            # xv  = torch.tensor(arry[:,1],dtype=torch.float32)
            xi, xv = [x for x in batch]
            logits = eval_model(xi,xv)
            probs  = torch.sigmoid(logits).detach().cpu().numpy().ravel().tolist()
            all_predictions.extend(probs)

    return pd.Series(all_predictions)


if __name__ =='__main__':
#     parser = argparse.ArgumentParser(description='reno5_pytorch_inference')
#     parser.add_argument('--model_state',type=str,required=True)
#     parser.add_argument('--config',type=str,required=True)
#     parser.add_argument('--features',type=str,required=True)
   
#     args = parser.parse_args()

#     feature_cfg = yaml.load(open(args.features,'r'))
#     all_cfg = yaml.load(open(args.config,'r'))
#     network_cfg = all_cfg['FM']


    pretrained_cfg  = PretrainedConfig(**all_cfg['self_attention']) # for AutoInt
    pretrained_cfg.initializer_range = network_cfg['init_norm_var']
    pretrained_cfg.hidden_size = network_cfg['embedding_size']

    model = AutoInt(pretrained_cfg, network_cfg, field_size = feature_cfg['field_dim'], feature_size=feature_cfg['feature_dim'])
    model.load_state_dict(torch.load(args.model_state, map_location = device))  #map_Location=torch.device('cpu')

    bc_model_state = sc.broadcast(model.state_dict())
    eval_model = get_predict_model(args)
    eval_model.eval()
    eval_model.to(device)
    print(eval_model.high_order_embedding(torch.tensor([0,1,202],dtype=torch.long)))
    print(eval_model.high_order_embedding.weight.shape)
    # print(eval_model)
    # print(eval_model.high_order_project.weight.data)

    
    feature_cols = feature_cfg['cat_cols'] + feature_cfg['double_cols']
    total_cols = ['imei'] + feature_cols 
    print(total_cols)
    cols = ','.join(total_cols)
#     df = spark.sql('select '+ cols+ ' from dataintel_tmp.churn_sample2pred_1217 where gender is not null')
    df = spark.sql('select {} from {} where gender is not null'.format(cols, args.pred_table))
    df = df.repartition(2000)



    ##---------------------------------- preprocess-----------------------------
    # print(df.head(2))
    #df = df.fillna('-999',subset=cat_cols)

    # df = df.select('imei', *[data_df[col].cast(DoubleType()) for col in input_features])
    print('preprocess'.center(50,'='))
    df = preprocess(df, feature_cfg)
    # df.printSchema()
    print('get_model_input'.center(50,'='))
    df = get_model_input(df, feature_cfg)
    # df.printSchema()
    
    print('inference'.center(50,'='))
    df = df.withColumn('prob',model_udf_inference(df.feats))
    df.printSchema()
    # df.write.mode("overwrite").saveAsTable("dataintel_tmp.reno5_lookalike_demo")
    # df.write.saveAsTable(save_table, mode='append', partitionBy=['pt_day'])


    df = df.select('imei','prob')
    df.persist()
    df.createOrReplaceTempView('tmp_table')
    spark.sql('create table if not exists {} as select * from tmp_table'.format(args.result_table))
    

    #==================================test=========================================
#     cols = ['imei','prob'] + feature_cols 
#     df = df.select(*(cols))
#     df.persist()
#     df.createOrReplaceTempView('tmp_table')
#     spark.sql('create table if not exists dataintel_tmp.reno4_lookalike_autoint  as select * from tmp_table')
    

    spark.stop()

#     df.printSchema()
#     SparkFiles.get(args.config)
