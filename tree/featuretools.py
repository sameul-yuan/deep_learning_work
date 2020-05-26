# -*- coding: utf-8 -*-
import featuretools as ft
#
#featuretools  is used for  transforming and aggregating information from one-to-many relationship

data = ft.demo.load_mock_customer()


customs = data['customers']
sessions = data['sessions']
transactions = data['transactions']
transactions = transactions.merge(sessions).merge(customs)
print(transactions.head())
products = data['products']
print(products.head())

#--------------define an empty entitySet----------------------
es =ft.EntitySet(id='demo')
#---------------add entity-----------------------------------
es =es.entity_from_dataframe(entity_id='transactions',
                             dataframe=transactions,
                             index='transaction_id',        # unique
                             time_index='transaction_time',
                             variable_types={'product_id':ft.variable_types.Categorical,
                                             'zip_code':ft.variable_types.ZIPCode}
                             )

es =es.entity_from_dataframe(entity_id='products',
                             dataframe=products,
                             index='product_id')


print(es)
print(es['transactions'].variables)
#---------------add relationship between entities in the entitySet---

relation1 = ft.Relationship(es['products']['product_id'],   # one - to - many
                            es['transactions']['product_id'])  # groupby(product_id)

es  =es.add_relationship(relation1)
print(es)
# print(es['transactions'].variables)

#------------------do deep feature synthesis(dfs)--------------------------
# feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='products')
feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='products',
                                      agg_primitives=['count'],  # aggregation function apply between entities
                                      trans_primitives=['month'], # transform function apply to target_entity
                                      max_depth=1)
print(feature_matrix.columns.tolist())
print(feature_matrix.head())
print(feature_defs)

print('-------seed feature( used defined feature) ---')
expansive_purchase = ft.Feature(es['transactions']['amount'])>100
feature_matrix, feature_defs = ft.dfs(entityset=es,
                                      target_entity='products',
                                      agg_primitives=['percent_true'],
                                      seed_features=[expansive_purchase])
print(feature_matrix.columns.tolist())
print(feature_matrix.head())
print(feature_defs)

print('---------where primitives-------')
es['transactions']['date_of_birth'].interesting_values=['1986-08-18','1986-08-19']  #'where_primitives' to specify agg primitives in agg_primitives
feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='products',
                                      where_primitives=['count'],
                                      agg_primitives=['count', 'mean'], # specified, otherwise defaults primitives will be used
                                      max_depth=1)
print(feature_matrix.columns.tolist())
print(feature_matrix.head())
print(feature_defs)

print('-----------encode category feature-----------')
feature_matrix_enc, feature_enc = ft.encode_features(feature_matrix,feature_defs)
print(feature_matrix_enc.columns.tolist())
print(feature_matrix_enc.head())
print(feature_enc)

print('-----------list primitives---------------------')
print(ft.list_primitives().head())

print('----------custom primitives----------------------')
from featuretools.primitives import make_agg_primitive,make_trans_primitive
from featuretools.variable_types import Text,Numeric
def absolute(column):
    return abs(column)
Absolute =make_trans_primitive(function=absolute,
                               input_types=[Numeric],
                               return_type=Numeric)
def maximum(columns):
    return max(columns)
Maximum = make_agg_primitive(function=maximum,
                             input_types=[Numeric],
                             return_type=Numeric)
#Multiple Input Types
def mean_numeric(num1, num2):
    return (num1+num2)/2

Meanval = make_trans_primitive(function=mean_numeric,
                               input_types=[Numeric,Numeric],
                               return_type=Numeric)
feature_matrix, feature_defs = ft.dfs(entityset=es, target_entity='transactions',
                                      trans_primitives=[Meanval], # specified, otherwise defaults primitives will be used
                                      max_depth=1,
                                      ignore_variables={'transactions': ['session_id']}, # exclude feature to be processed in  trans/agg primitives
                                      max_features=-1) # cut the  first `max_features` to return
print(feature_matrix.columns.tolist())
print(feature_matrix.head())
print(feature_defs)
print(feature_matrix.dtypes)