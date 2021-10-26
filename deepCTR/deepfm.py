import os
import sys
import itertools
import numpy as np
import torch
import torch.nn.functional as  F

# 

class DeepFM(torch.nn.Module):
    """
    field_size: F
    feature_size: N
    embedding_size: K

    """
    def __init__(self, params_config, field_size=None, feature_size=None):
        super(DeepFM, self).__init__()
        
        # self.feat_cfg = feature_config
        self.field_size = field_size
        self.feature_size = feature_size
        self.emb_size = params_config.pop('embedding_size')
        self.param_cfg = params_config


        self.linear_w = torch.nn.Embedding(self.feature_size,1)  # F*1 
        self.embedding = torch.nn.Embedding(self.feature_size, self.emb_size) # F*K
        
        # attention param for second interaction
        if self.param_cfg['use_afm']:
            self.attention_size = self.param_cfg['attention_size']
            self.attention_linear = torch.nn.Linear(self.emb_size, self.attention_size,bias=True) # f.sum(aij.<vi,vj>xi.xj)
            self.attention_h = torch.nn.Linear(self.attention_size,1,bias=False)
            self.attention_f = torch.nn.Linear(self.emb_size, 1, bias=False)
            if self.param_cfg['attention_drop_p']:
                self.attention_drop = torch.nn.Dropout(p=self.param_cfg['attention_drop_p'])
            
            self.attention_linear.bias.data.fill_(0)
            torch.nn.init.kaiming_normal_(self.attention_linear.weight.data)
            torch.nn.init.normal_(self.attention_h.weight, std=self.param_cfg['init_norm_var'])
            torch.nn.init.normal_(self.attention_f.weight, std=self.param_cfg['init_norm_var'])

            
        # dnn layers
        deep_layers = params_config['hidden_layer_sizes']
        input_size = self.field_size * self.emb_size # FK
        deep_layers.insert(0, input_size)
        deep_layers.append(1)  
 
        
        self.deep_model  = torch.nn.Sequential()
        for idx,(ind, od)in enumerate(zip(deep_layers[:-1],deep_layers[1:])):  #[input, h0,h1,h2,1]
            if idx !=len(deep_layers[1:])-1 :
                self.deep_model.add_module('linear_{}'.format(idx), torch.nn.Linear(ind,od))
                self.deep_model.add_module('batchnrom_{}'.format(idx), torch.nn.BatchNorm1d(od))
                self.deep_model.add_module('acitive_{}'.format(idx), torch.nn.ReLU())
                self.deep_model.add_module('dropout_{}'.format(idx),torch.nn.Dropout(self.param_cfg['drop_prob'])) # NOTE batchnorm + dropout
            else:
                self.deep_model.add_module("linear_{}".format(idx),torch.nn.Linear(ind,od))
        
        # last_layer_input_size = self.field_size + self.emb_size + deep_layers[-1]
        # self.last_layer = torch.nn.Linear(last_layer_input_size, 1)
        # torch.nn.init.xavier_normal_(self.last_layer.weight.data)
        # self.last_layer.bias.data.zero_()

        #--init---
        self.linear_w.apply(self.__init_weight)
        self.embedding.apply(self.__init_weight)
        self.deep_model.apply(self.__init_weight)


    def __init_weight(self, module):
        if isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(0, self.param_cfg['init_norm_var']) #  nn.init.normal_(module.weight.data,0,0.02),module.weight.data.fill_(0)/zero_()
            
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight.data)
            module.bias.data.normal_(std=self.param_cfg['init_norm_var'])


    def forward(self, xi,xv):
        """
        xi: (bsz,F)
        xv: (bsz,F)

        """

        # linear_part
        first_emb = self.linear_w(xi.long()) # (bsz, F, 1)
        first_emb = torch.mul(first_emb, torch.unsqueeze(xv,-1)).float() #(bsz,F,1)
        first_order_out = torch.sum(first_emb,dim=(1))  #(bsz, 1)
        # second_part
        embs = self.embedding(xi.long())  # (bsz,F,K)
        embs = torch.mul(embs, torch.unsqueeze(xv,-1)).float()
        
        if self.param_cfg['use_afm']:  # FM with attention
            index1,index2 = list(zip(*itertools.combinations(range(self.field_size),2)))
            p = embs[:,index1]    #(bsz, F(F-1)/2, K)
            q = embs[:,index2]
            inner_product  = p*q  #(bsz,F(F-1)/2, K)

            if self.param_cfg['attention_drop_p']:
                inner_product = self.attention_drop(inner_product)
            att_tmp  = self.attention_linear(inner_product) #(bsz, F(F-1)/2, attention_size)
            att_tmp  = torch.nn.functional.relu(att_tmp)
            att_score  = self.attention_h(att_tmp) #(bsz, K(K-1)/2, 1)
            att_score = torch.nn.functional.softmax(att_score, dim=1)

            att_value = torch.sum(inner_product*att_score, dim=1) #(bsz,K)
            second_order_out = self.attention_f(att_value) # (bsz,1)

        else: # general FM 
            sum_second_order = torch.sum(embs,dim=1) #(bsz,K)
            sum_second_order_square = torch.pow(sum_second_order, 2) 
            square_second_order = torch.pow(embs,2)
            square_second_order_sum = torch.sum(square_second_order,dim=1) #(bsz,K)
            second_order_out = 0.5*torch.sum(sum_second_order_square-square_second_order_sum,dim=1,keepdim=True) #(bsz,1)
        
        # # print(first_order_out.dtype,second_order_out.dtype)
        # # fm_out = torch.cat([first_order_out,second_order_out],dim=1) #(bsz,F+K)

        # # DNN
        dnn_input = embs.view(-1, self.field_size*self.emb_size) #(bsz,F*K)
        dnn_out = self.deep_model(dnn_input)  #(bsz, 1)

        # # last_input = torch.cat([fm_out, dnn_out],dim=1) #(bsz,F+K+H)
        # # last_out = self.last_layer(last_input)  #(bsz,1)
        # last_out = first_order_out+second_order_out+dnn_out

        last_out = first_order_out+second_order_out+dnn_out
        return last_out



if __name__ == '__main__':
    import pandas as pd 
    import json
    import yaml
    from torch.utils.data import SequentialSampler, DataLoader
    from dataset import DeepFmDataset
    from torchviz import make_dot

    data_path = '../data/reno2train_transformed_with_emb_log.pkl'
    feature_cfg  = json.load(open('../server/DataNumerify_with_emb_log.dict','r'))
    param_cfg = yaml.load(open('../config.yaml','r'))
    param_cfg = param_cfg['FM']

    df = pd.read_pickle(data_path)
    print(df.shape)
    df.reset_index(drop=True, inplace=True)

    dataset = DeepFmDataset(df,feature_cfg)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=16)

    model = DeepFM(param_cfg,field_size=feature_cfg['field_dim'],feature_size=feature_cfg['feature_dim'])

    for idx, batch in enumerate(dataloader):
        xi,xv,label=batch
        model.eval()
        logits = model(xi,xv)
        if idx==0:
            print(logits)
            break  
    
    # print(len(dataloader))
    # g = make_dot(logits, params=dict(list(model.named_parameters()) + [('xi', xi),('xv',xv)]))
    # g.render('../deepfm', view=False) 

    



      







        
        







