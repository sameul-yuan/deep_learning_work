# export SPARK_HOME=/usr/local/spark-2.4.0-bin-hadoop2.6
# export SPARK_HOME=/usr/local/share/spark
export SPARK_HOME=/usr/local/spark
export HADOOP_USER_NAME=xxxxxx

ALG_HDFS=hdfs://xxx-hdfs
python_path=./env/spark_env2/bin/python

${SPARK_HOME}/bin/spark-submit \
--queue root.default \
--name "spark_torch_inference_submit" \
--master yarn \
--driver-memory 40g \
--executor-memory 25g \
--deploy-mode cluster \
--num-executors 200 \
--executor-cores 8 \
--conf spark.port.maxRetries=1 \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.executor.memoryOverhead=10g \
--conf spark.shuffle.memoryFraction=0.4 \
--conf spark.default.parallelism=3000 \
--conf spark.pyspark.python=$python_path \
--conf spark.pyspark.driver.python=$python_path \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=$python_path \
--conf spark.driver.maxResultSize=10g \
--conf spark.hadoop.fs.defaultFS=$ALG_HDFS \
--conf spark.sql.execution.arrow.pyspark.enabled=true \
--archives /home/notebook/code/personal/auto_train_deep/spark_env2.zip#env \
--files model.dict,_config_.yaml,features.dict \
--py-files autoint_pyspark.py \
spark_inference_auto.py --model_state model.dict --config _config_.yaml --features features.dict --pred_table ${1} --result_table ${2}

# /home/notebook/code/public/ogp/pyenv/py37.zip#py37 

# hdfs://alg-hdfs/livydata/xxxvirenvpython37.tar.gz

##--archives  hdfs://data-batch-hdfs/user/xxxx/envs/env_torch12.tar.gz#py37 \

# /home/notebook/code/group/smart_rec/env_torch12.tar.gz#py37 \
# spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "64")

# spark.driver.maxResultSize=10G

#  config('spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT',1)
# config('spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT',1)
