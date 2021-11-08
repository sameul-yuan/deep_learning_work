#encoding: utf-8
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nets import AbstractModel, DNN, CrossNet, PredictionHead
from .inputs import (SparseFeat, DenseFeat, LinearLogits, get_feature_names, compute_input_dim, build_input_features,
                    create_embedding_matrix, combined_dnn_input, input_from_feature_columns)


class DCNModel(nn.Module):
    def __init__(self, param_conf,feature_conf):
        super().__init__()
        self.cross_num = param_conf['cross_num']
        self.cross_parameterization = param_conf['cross_parameterization']
        self.dnn_hidden_units = param_conf['dnn_hidden_units']
        self.activation = param_conf['activation']
        self.task = param_conf['task']
#         print(param_conf)
        # feature
        self.sparse_feature_size = feature_conf['sparse_feature_size']

        self.sparse_feature_columns = [
               SparseFeat(feat, vocabulary_size=self.sparse_feature_size[feat], embedding_dim=param_conf['embedding_dim'])
                           for feat in feature_conf['sparse_feature_columns'] ]
        self.dense_feature_columns = [DenseFeat(feat, dimension=1) for feat in feature_conf['dense_feature_columns']]
        
        self.dense_emb_feature_columns = []
        
        # feature 和 dataFrame列的对应关系 sparse + dense + varLen, 
        self.input_feature_columns = self.sparse_feature_columns + self.dense_feature_columns + self.dense_emb_feature_columns
        self.feature_index = build_input_features(self.input_feature_columns)  # Ordereddict(featname:start,end), Dataset的返回顺序需要和这个一致(dense和denseEmb需要避免重复)
        

        # embedding
        if len(self.dnn_hidden_units)>0 or self.cross_num>0:
            self.embedding_dict = create_embedding_matrix(self.input_feature_columns, include_dense_emb=False) # {featrue_name: nn.embedding}

        # Deep
        if len(self.dnn_hidden_units)>0:
            self.dnn = DNN(compute_input_dim(self.input_feature_columns), self.dnn_hidden_units, activation=self.activation) # compute_input_dim 输入维度

        # Cross
        if self.cross_num>0:
            self.crossnet = CrossNet(in_features=compute_input_dim(self.input_feature_columns),  #选择需要cross的特征
                                     layer_num=self.cross_num,
                                     parameterization=self.cross_parameterization)
        # Linear
        dnn_linear_input_dim = 0 # dense layer + cross layer拼接
        if len(self.dnn_hidden_units) > 0 :
            dnn_linear_input_dim += self.dnn_hidden_units[-1]
        if self.cross_num>0:
            dnn_linear_input_dim += compute_input_dim(self.input_feature_columns) 
            
        self.dnn_linear = nn.Linear(dnn_linear_input_dim, 1, bias=False)  
        # LR layer
#         self.logit = LinearLogits(self.input_feature_columns, self.feature_index)
#         self.bias = nn.Parameter(torch.zeros((1, )))
        self.bias = 0

        #self.out = PredictionHead(self.task, use_bias=True) # logits to prob

    def forward(self, x):
        # x = batch['x'].float()
#         logit = self.logit(x) #LR 
        logit = 0
        # x 和 self.feature_index的顺序要匹配，embdding_list中的序列和input_feature_columns顺续相关
        sparse_embedding_list, dense_value_list = input_from_feature_columns(x, self.input_feature_columns,
                                                                             self.feature_index, self.embedding_dict) #根据feature_columns 匹配data中的数据列

        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list) #concat_embedding 

        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:  # Deep & Cross
            deep_out = self.dnn(dnn_input)
            cross_out = self.crossnet(dnn_input)
            stack_out = torch.cat((cross_out, deep_out), dim=-1)
            logit += self.dnn_linear(stack_out)

        elif len(self.dnn_hidden_units) > 0:  # Only Deep
            deep_out = self.dnn(dnn_input)
            logit += self.dnn_linear(deep_out)
        elif self.cross_num > 0:   # Only Cross
            cross_out = self.crossnet(dnn_input)
            logit += self.dnn_linear(cross_out)
        else:  # only LR
            print('LR')
            
        logit += self.bias
        return logit
    

# class DCNModel(nn.Module):
#     def __init__(self, param_conf,feature_conf):
#         super().__init__()
#         self.cross_num = param_conf['cross_num']
#         self.cross_parameterization = param_conf['cross_parameterization']
#         self.dnn_hidden_units = param_conf['dnn_hidden_units']
#         self.activation = param_conf['activation']
#         self.task = param_conf['task']

#         # feature
#         self.sparse_feature_size = feature_conf['sparse_feature_size']

#         self.sparse_feature_columns = [
#                SparseFeat(feat, vocabulary_size=self.sparse_feature_size[feat], embedding_dim=param_conf['embedding_dim'])
#                            for feat in feature_conf['sparse_feature_columns'] ]
#         self.dense_feature_columns = [DenseFeat(feat, dimension=1) for feat in feature_conf['dense_feature_columns']]

#         self.input_feature_columns = self.sparse_feature_columns + self.dense_feature_columns
#         self.feature_index = build_input_features(self.input_feature_columns) # feature 和dataFrame 列的对应关系 sparse + dense + varLen
#         # embedding
#         self.embedding_dict = create_embedding_matrix(self.input_feature_columns)

#         # Deep
#         self.dnn = DNN(compute_input_dim(self.input_feature_columns), self.dnn_hidden_units, activation=self.activation) # compute_input_dim 输入维度

#         # Cross
#         self.crossnet = CrossNet(in_features=compute_input_dim(self.input_feature_columns),  #选择需要cross的特征
#                                  layer_num=self.cross_num,
#                                  parameterization=self.cross_parameterization)

#         # Linear
#         if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
#             dnn_linear_in_feature = compute_input_dim(self.input_feature_columns) + self.dnn_hidden_units[-1] #cross layer + dense layer 拼接
#         elif len(self.dnn_hidden_units) > 0:
#             dnn_linear_in_feature = self.dnn_hidden_units[-1]
#         elif self.cross_num > 0:
#             dnn_linear_in_feature = compute_input_dim(self.input_feature_columns)

#         self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False)  #bias在predictionHead 加
#         # LR layer
#         self.logit = LinearLogits(self.input_feature_columns, self.feature_index)
#         self.bias = nn.Parameter(torch.zeros((1, )))
# #         self.out = PredictionHead(self.task, use_bias=True) # logits to prob

#     def forward(self, x):
#         # x = batch['x'].float()
# #         x = batch
#         logit = self.logit(x) #LR 
# #         logit = 0
#         sparse_embedding_list, dense_value_list = input_from_feature_columns(x, self.input_feature_columns,
#                                                                              self.feature_index, self.embedding_dict) #匹配data中的数据列

#         dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list) #concat_embedding 

#         if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:  # Deep & Cross
#             deep_out = self.dnn(dnn_input)
#             cross_out = self.crossnet(dnn_input)
#             stack_out = torch.cat((cross_out, deep_out), dim=-1)
#             logit += self.dnn_linear(stack_out)

#         elif len(self.dnn_hidden_units) > 0:  # Only Deep
#             deep_out = self.dnn(dnn_input)
#             logit += self.dnn_linear(deep_out)
#         elif self.cross_num > 0:   # Only Cross
#             cross_out = self.crossnet(dnn_input)
#             logit += self.dnn_linear(cross_out)
#         else:  # only LR
#             print('LR')
            

#         logit += self.bias
#         return logit
          

if __name__=="__main__":
    import yaml
    import json
    import os
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_conf =yaml.load(open(os.path.join(base_path,'_config_.yaml'),'r'))['DCN']
    feature_conf = yaml.load(open(os.path.join(base_path,'tmp/features_20211026.dict'),'r'))
    model=DCNModel(param_conf,feature_conf)
    print(model)
    
