#!/bin/bash
# Program:
# History:
# 20210809    80261293      First  release    # 保险 数据集的多目标任务学习模型
#########################################################################
###全局变量定义和引入
#########################################################################
SCRIPT_PATH=/data1/etl_sys/script
COMMON_PATH=/data1/etl_sys/common
source ${COMMON_PATH}/export.ini
source ${COMMON_PATH}/common.sh
source ${COMMON_PATH}/DayGen.ini $1
source ${COMMON_PATH}/spark_livy.sh
#########################################################################
### 私有变量定义和引入
#########################################################################
v_job_stat=0

feat_tbl='loan_algo.f_evt_ins_app_field_feature_d'
score_tbl='loan_algo.f_evt_insurance_model_deepfm_click_v1'

#########################################################################
### 主程序
#########################################################################

#########################################################################
### 写表
#########################################################################
hql="""
create table if not exists ${score_tbl}(
    imei         string    comment  'imei',
    score        float    comment  'score'
)
comment '保险-deepfm-点击率模型v1'
partitioned by(
    dayno               string
)
STORED AS ORC
;
"""
#echo $(date +%Y-%m-%d:%T) "$hql"
#ExecuteSparkSqlOnLivy 'hive' "${hql}" "${feat_tbl}-${v_day}_1"
#v_job_stat=`expr ${v_job_stat} + $?`

#########################################################################
###提交任务
#########################################################################
/usr/local/share/spark/bin/spark-submit \
--queue datamin.default \
--name "${score_tbl}_${v_day}" \
--master yarn \
--driver-memory 20g \
--executor-memory 50g \
--deploy-mode cluster \
--num-executors 200 \
--executor-cores 2 \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.sql.shuffle.partitions=1000 \
--conf spark.io.compression.codec=lz4 \
--conf spark.shuffle.blockTransferService=nio \
--conf spark.executor.cores=3 \
--conf spark.driver.maxResultSize=0 \
--conf spark.rpc.message.maxSize=500 \
--conf spark.executor.memoryOverhead=8192 \
--conf spark.speculation=true \
--conf spark.speculation.interval=10000 \
--conf spark.speculation.quantile=0.6 \
--conf spark.network.timeout=600 \
--conf spark.core.connection.ack.wait.timeout=1000 \
--conf spark.shuffle.consolidateFiles=true \
--conf spark.default.parallelism=1000 \
--conf spark.port.maxRetries=10 \
--conf spark.sql.execution.arrow.maxRecordsPerBatch=500 \
--conf "spark.pyspark.driver.python=/usr/local/anaconda3/bin/python3.6" \
--conf "spark.pyspark.python=/usr/local/anaconda3/bin/python3.6" \
--files ${SCRIPT_PATH}/f_evt_insurance_model_deepfm_click_v1.pt,${SCRIPT_PATH}/ins_merge_field2onehot_app.txt \
${SCRIPT_PATH}/f_evt_insurance_model_deepfm_click_v1.py ${feat_tbl} ${score_tbl} ${v_day}
v_job_stat=`expr ${v_job_stat} + $?`


#########################################################################
### 更新作业完成标志，若失败则读取错误信息。
#########################################################################
echo "v_job_stat is $v_job_stat"
exit ${v_job_stat}
