import os
import re
import sys
import pandas as pd
import numpy as np
from datetime import datetime 
from IPython.core.interactiveshell import InteractiveShell 
InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_rows',100)

submit_args = """
--queue datamin.high
--master yarn
--deploy-mode client
--driver-memory 20g
--driver-cores 3
--num-executors 25
--executor-memory 30g
--conf spark.executor.cores=3
--conf spark.yarn.executor.memoryOverhead=8384
--conf spark.dynamicAllocation.maxExecutors=50 
--conf spark.dynamicAllocation.enabled=true 
--conf spark.default.parallelism=1000
--conf spark.sql.shuffle.partitions=600
--conf spark.shuffle.blockTransferService=nio
--conf spark.driver.maxResultSize=300m
--conf spark.rpc.message.maxSize=500
--conf spark.speculation=true
--conf spark.speculation.interval=10000
--conf spark.speculation.quantile=0.6
--conf spark.network.timeout=300
--conf spark.kryoserializer.buffer.max.mb=1024
--conf spark.core.connection.ack.wait.timeout=300
--conf spark.shuffle.consolidateFiles=true
--conf spark.sql.execution.arrow.pyspark.enabled=true 
--conf spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT=1
--conf spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT=1
pyspark-shell
"""
os.environ['PYSPARK_SUBMIT_ARGS'] = submit_args
os.environ['SPARK_HOME'] = '/usr/local/share/spark'
os.environ['ARROW_PRE_0_15_IPC_FORMAT'] = '1'

os.environ["SPARK_PYTHONPATH"] = '/usr/local/anaconda3/envs/spark_env/bin/python3.6' #driver对应的python版本
os.environ["PYSPARK_PYTHON"] = '/usr/local/anaconda3/bin/python3.6'  # worker对应的python版本


import findspark
findspark.init()

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("finance_coupon") \
    .config("spark.files.overwrite", "true") \
    .config("hive.support.quoted.identifiers", None) \
    .getOrCreate()
sc = spark.sparkContext

# from pyspark.sql.functions import udf, desc, asc
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType, StringType, DoubleType, IntegerType, StructType, StructField,DateType
from pyspark.sql.window import Window

# from pyspark.sql.functions import broadcast,date_sub,date_format,datediff,pandas_udf,PandasUDFType

import calendar
import os
import logging
import json
import sys
import glob
import gc
import argparse
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import  relativedelta
from scipy import sparse,stats
from collections import Counter, OrderedDict
import multiprocessing

import matplotlib.pyplot as plt
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from utils import get_auc_ks,calc_threshold_vs_depth,encode_category_feature  #calc_feature_importance

#pandas_UDF, 有两种类型的Pandas_UDF，分别是Scalar（标量映射）和Grouped Map（分组映射）

#=================1. Scalar Pandas UDF用于向量化标量操作。常常与select和withColumn等函数一起使用。其中调用的Python函数需要使用pandas.Series作为输入并返回一个具有相同长度的pandas.Series。具体执行流程是，Spark将列分成批，并将每个批作为数据的子集进行函数的调用，进而执行panda UDF，最后将结果连接在一起。
# 声明函数并创建UDF
 
def multiply_func(a, b):
    return a * b
multiply = pandas_udf(multiply_func, returnType=LongType())
x = pd.Series([1, 2, 3])
df = spark.createDataFrame(pd.DataFrame(x, columns=["x"]))
 
# Execute function as a Spark vectorized UDF
df.select(multiply(col("x"), col("x"))).show()
# +-------------------+
# |multiply_func(x, x)|
# +-------------------+
# |                  1|
# |                  4|
# |                  9|
# +-------------------+

#===============2. Group Map， （分组映射）panda_udf与groupBy().apply()一起使用,StructType对象中的Dataframe特征顺序需要与分组中的Python计算函数返回特征顺序保持一致
df = spark.createDataFrame(
    [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
    ("id", "v"))
 
@pandas_udf("id long, v double", PandasUDFType.GROUPED_MAP)
def subtract_mean(pdf):
    # pdf is a pandas.DataFrame
    v = pdf.v
    return pdf.assign(v=v - v.mean())
 
df.groupby("id").apply(subtract_mean).show()
# +---+----+
# | id|   v|
# +---+----+
# |  1|-0.5|
# |  1| 0.5|
# |  2|-3.0|
# |  2|-1.0|
# |  2| 4.0|
# +---+----+

#3. =============== group agg :F常常与groupBy().agg()和pyspark.sql.window一起使用
from pyspark.sql import Window
 
df = spark.createDataFrame(
    [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
    ("id", "v"))
 
@pandas_udf("double", PandasUDFType.GROUPED_AGG)
def mean_udf(v):
    return v.mean()
 
df.groupby("id").agg(mean_udf(df['v'])).show()
# +---+-----------+
# | id|mean_udf(v)|
# +---+-----------+
# |  1|        1.5|
# |  2|        6.0|
# +---+-----------+

w = Window \
    .partitionBy('id') \
    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
df.withColumn('mean_v', mean_udf(df['v']).over(w)).show()
# +---+----+------+
# | id|   v|mean_v|
# +---+----+------+
# |  1| 1.0|   1.5|
# |  1| 2.0|   1.5|
# |  2| 3.0|   6.0|
# |  2| 5.0|   6.0|
# |  2|10.0|   6.0|
# +---+----+------+

