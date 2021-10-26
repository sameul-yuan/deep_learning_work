import copy
import yaml
import json
import os
import sys
import random
import logging
import time
import pandas as pd
import numpy as np 
from datetime import datetime
from sklearn.metrics import auc,roc_auc_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler, DistributedSampler
# from torch.utils.tensorboard import SummaryWriter


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import DeepFmDataset
from src.deepfm import DeepFM
from src.utils import EarlyStopping, calc_threshold_vs_depth
from src.losses import BCEFocalLoss

# from src.self_attention.config import PretrainedConfig
# from src.autoint import AutoInt, get_linear_schedule_with_warmup

from src.autoint_pyspark import AutoInt, get_linear_schedule_with_warmup, PretrainedConfig


import atexit
import subprocess as spc

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s： %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(seed, n_gpu=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu>0:
        torch.cuda.manual_seed_all(seed)


def train(model, train_dataset, eval_dataset, checkpoint_dir,
         writer=None,
         early_stop_patience=4,
         lr_schedule_patience=3,
         learning_rate = 1e-3,
         min_lr=1e-5,
         lr_factor= 0.2,
         n_gpu=1,
         per_gpu_batch_size=32,
         gradient_accum_steps=1,
         device='cpu',
         weight_decay=1.0e-5,
         max_steps=100000,
         warmup_steps=5000,
         epochs = 30,
         start_epoch=0,
         pos_weight=1.0,
         max_grad_norm = 1.0,
         seed = 42,
         eval_steps = 5000,
         logging_steps = 500,
         **kwargs):

    early_stop = EarlyStopping(patience=early_stop_patience)

    #设置dataloader
    n_gpu = max(1,min(torch.cuda.device_count(), n_gpu))
    train_batch_size = per_gpu_batch_size * n_gpu

    sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=train_batch_size, num_workers=2)

    # 设置优化器
    model.to(device) # before allocated to multi-gpu, model should be loaded into logical gpu[0]
    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=lr_schedule_patience, min_lr=min_lr,factor=lr_factor)
    # scheduler = get_linear_schedule_with_warmup(optimizer,warmup_steps,max_steps)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    #多GPU
    if n_gpu > 1: 
        print('DataParallel model')
        model = torch.nn.DataParallel(model)
    logger.info('***********start running************')
    logger.info('batch_size：{}'.format(train_batch_size))
    logger.info('max_steps:{}'.format(max_steps))

    global_step = start_epoch
    step_eval_loss = np.Inf
    epoch_eval_loss = np.Inf
    best_auc = 0
    
    #初始化  (torch.tensor(app_ids), torch.tensor(time_ids),  torch.tensor(duration_ids), torch.tensor(lengths), torch.tensor(labels))
    model.zero_grad()
    set_seed(seed)

    use_focalloss = kwargs.get('use_focalloss',False)

    if use_focalloss:
        alpah = kwargs.get('focal_alpha',0.25)
        gamma = kwargs.get('focal_gamma',2)
        loss_fct = BCEFocalLoss(gamma=gamma,alpha=alpah)
    else:
        loss_fct =  F.binary_cross_entropy_with_logits

   
    for idx in range(start_epoch, epochs): 
        logger.info(('epoch={}'.format(idx)).center(20,'-'))
        for _, batch in enumerate(train_dataloader):

            optimizer.zero_grad()

            xi,xv,label = [x.to(device) for x in batch]
            model.train() 
            logits = model(xi,xv)
           
            if use_focalloss:
                # loss1 = loss_fct(logit1,label)
                # loss2 = loss_fct(logit2,label)
                # loss3 = loss_fct(logit3,label)
                loss = loss_fct(logits,label)
                # loss = loss + 0.2*loss1 + 0.2*loss2 + 0.2*loss3
            else:
                tensor_pos_weight = torch.where(label==1, torch.tensor(pos_weight).float().to(device), torch.tensor(1.0).to(device))
                loss = loss_fct(logits, label, pos_weight=tensor_pos_weight)

            if n_gpu>1:
                loss = loss.mean() #
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            # scheduler.step()
            
            # train logger 
            if global_step % logging_steps==0: 
                print("step={:d},loss={:.4f}".format(global_step,  loss.item()))
            global_step += 1    

        # evaluate after each opoch
        train_result = evaluate(model, train_dataset, device=device, batch_size=per_gpu_batch_size, use_focalloss=use_focalloss, loss_fct=loss_fct,  pos_weight=pos_weight)
        eval_result = evaluate(model, eval_dataset, device=device, batch_size=per_gpu_batch_size, use_focalloss=use_focalloss, loss_fct=loss_fct,pos_weight=pos_weight)
        scheduler.step(eval_result['eval_loss'])
        # scheduler.step()
                
        logger.info('epoch={:d}, train:{}, evaluation: {}\n'.format(idx, train_result, eval_result))  
        if writer:
            # writer.add_scalars('losses',{'train_loss':train_result['eval_loss'],'eval_loss':eval_result['eval_Loss']},global_step=global_step)
            # writer.add_scalars('auc',{'train_auc':train_result['auc'],'eval_auc':eval_result['auc']},global_step=global_step)
            writer.add_scalar('eval_loss',eval_result['eval_loss'], global_step=global_step)
            writer.add_scalar('eval_auc',eval_result['auc'], global_step=global_step)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            writer.add_scalar('train_loss',train_result['eval_loss'], global_step=global_step)
            writer.add_scalar('train_auc',train_result['auc'], global_step=global_step)

            # writer.add_scalars('logits',{'first_order':logit1.item(),'second_order': logit2.item(),'high_order':logit3.item()},global_step=global_step)
            # writer.add_histogram('emb',model.high_order_embedding.weight.grad.cpu(),global_step=global_step)
            # writer.add_histogram('logit1',logit1.data.cpu(), global_step=global_step)
            # writer.add_histogram('logit2',logit2.data.cpu(), global_step=global_step)
            # writer.add_histogram('logit3',logit3.data.cpu(), global_step=global_step)

        
        if eval_result['eval_loss'] < epoch_eval_loss:
            epoch_eval_loss = eval_result['eval_loss']
            
            best_auc= round(eval_result['auc'], 4)
            logger.info('update_epoch-{}_model'.format(idx))
            
            checkpoint_last = os.path.join(checkpoint_dir,'checkpoint_last')
            if not os.path.exists(checkpoint_last):  
                os.makedirs(checkpoint_last, mode=0o777, exist_ok=False)

            logger.info('saving model checkpoint to ...{}'.format(checkpoint_last))
            model_to_save = model.module if hasattr(model,'module') else model
            torch.save(model_to_save,os.path.join(checkpoint_last,'model.pt'))
            torch.save(model_to_save.state_dict(), os.path.join(checkpoint_last,'model.dict'))

        early_stop(eval_result['eval_loss'])
        if early_stop.early_stop or global_step>=max_steps or idx>epochs: 
            signature = 'early_stop' if early_stop.early_stop else 'max-steps-done'
            logging.info(('{}'.format(signature)).center(50,'*'))
            break

    return best_auc

def evaluate(model, eval_dataset, 
            batch_size=128, 
            n_gpu=1, 
            device='cpu',
            use_focalloss=False, 
            loss_fct=None,
            pos_weight=1.0,
            verbose=False): 
    eval_batch_size = batch_size * max(1, n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size, num_workers=2)

    eval_loss = 0.0
    eval_steps = 0 
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            xi, xv, label = [x.to(device) for x in batch]
            # x = torch.cat([xi.float(),xv],dim=1)
            logits = model(xi,xv)
            # logits= model(xi,xv)
            if use_focalloss:
                loss = loss_fct(logits,label)
            else:
                tensor_pos_weight = torch.where(label==1, torch.tensor(pos_weight).float().to(device), torch.tensor(1.0).to(device))
                loss = F.binary_cross_entropy_with_logits(logits, label, pos_weight=tensor_pos_weight)

            if n_gpu>1:  
                loss = loss.mean()

            eval_loss += loss.item()
            eval_steps +=1

            y_true.append(label)
            y_pred.append(torch.sigmoid(logits))

    eval_loss = eval_loss/eval_steps
    y_true = torch.cat(y_true,dim=0).squeeze().cpu().numpy()
    y_pred = torch.cat(y_pred,dim=0).squeeze().cpu().numpy()
    auc = roc_auc_score(y_true,y_pred)

    result = {'eval_loss': eval_loss, 'auc':auc}

    if verbose:
        result.update({'label':y_true,'pred':y_pred})
        
    return result


def predict(model, eval_dataset, 
            batch_size=128, 
            n_gpu=1, 
            device='cpu',
            use_focalloss=False, 
            loss_fct=None,
            pos_weight=1.0,
            verbose=False): 
    eval_batch_size = batch_size * max(1, n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size, num_workers=2)

    y_true = []
    y_pred = []
    all_probs = []
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            xi, xv, label = [x.to(device) for x in batch]
            logits = model(xi,xv)
            probs = torch.sigmoid(logits)
            # logits= model(xi,xv)

            all_probs.append(probs.detach().cpu().numpy())
    
    all_probs = np.concatenate(all_probs)

    df = pd.DataFrame(all_probs,columns=['prob'])
        
    return df


if __name__ == '__main__':

#     parser = argparse.ArgumentParser(description='train_model')
#     parser.add_argument('--raw_data',type=str,required=False)
#     args = parser.parse_args()
    

    is_predict = False
    # print('test')
    base_path = '/home/notebook/code/personal/auto_train_deep'
    
    all_cfg = yaml.load(open('./_config_.yaml'),'r')

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  #torch.cuda.device_count()


    # NOTE 2. for orginal_feature(400dim) + emb

    # data_path = os.path.join(base_path,'data/lookalike_autodis_log_level2_reduce_neg.pkl')
    # feature_cfg  = json.load(open(os.path.join(base_path,'server/lookalike_autodis_log_level2_reduce_neg.dict'),'r')) 

    data_path = all_cfg['tmp']['data_pkl_file']
    feature_cfg =  all_cfg['tmp']['feat_dict_file']
    signature = os.path.basename(data_path).split('.')[0]+'_'+os.path.basename(feature_cfg).split('.')[0]



    df = pd.read_pickle(data_path)
    drop_cols = ['label']
    df = df.drop(columns=drop_cols)
    df = df.rename(columns={'label_2m':'label'})
    print(df.shape)
    

    
    if not is_predict:  # for training model 
        network_cfg = all_cfg['FM']
        train_cfg = all_cfg['train']
        train_cfg['device'] = device

        pretrained_cfg  = PretrainedConfig(**all_cfg['self_attention']) # for AutoInt
        pretrained_cfg.initializer_range = network_cfg['init_norm_var']
        pretrained_cfg.hidden_size = network_cfg['embedding_size']

        signature = signature.format(network_cfg['embedding_size'], train_cfg['per_gpu_batch_size'],
                                  int(network_cfg['use_afm']),train_cfg['learning_rate'])

        # TODO(分层抽样)

        df_test = df.sample(frac=0.1, replace=False)
        df = df[~df.imei.isin(df_test.imei)]

        df_train, df_eval = train_test_split(df, stratify=df['label'], test_size=0.2, random_state=train_cfg['seed'])
     
        # df.reset_index(drop=True, inplace=True)
        # df_train = df.sample(frac=0.8,replace=False, random_state=train_cfg['seed'])
        # df_eval = df[~df.index.isin(df_train.index)]

        # df_train_pos = df_train[df_train.label==1]
        # df_train_pos = df_train_pos.sample(frac=3, replace=True)
        # df_train_neg = df_train[df_train.label==0]
        # df_train = pd.concat([df_train_pos,df_train_neg],axis=0,ignore_index=True)

        print(df_train.shape, df_eval.shape)
        print(df_train['label'].mean(), df_eval['label'].mean())

        train_dataset = DeepFmDataset(df_train,feature_cfg)
        eval_dataset = DeepFmDataset(df_eval,feature_cfg)
        
        checkpoint_dir = os.path.join(base_path,'model',time.strftime('%Y%m%d',time.localtime())+signature)
        logger.info(checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, mode=0o777, exist_ok=False)
        json.dump(all_cfg, open(os.path.join(checkpoint_dir,'config.json'),'w+'))
        # writer = SummaryWriter(log_dir=os.path.join(base_path,'log',signature),filename_suffix=signature)
        # atexit.register(writer.close)
        writer=None
        
        # model = DeepFM(network_cfg, field_size=feature_cfg['field_dim'], feature_size=feature_cfg['feature_dim'])
        model = AutoInt(pretrained_cfg, network_cfg, field_size=feature_cfg['field_dim'], feature_size=feature_cfg['feature_dim'])

        best_auc = train(model, train_dataset, eval_dataset, checkpoint_dir, writer=writer, **train_cfg)

        target_dir = checkpoint_dir+'_{}'.format(round(best_auc,4))
        spc.call(['mv '+checkpoint_dir+' ' + target_dir],shell=True)


        ##-------------------predict-------------------------------------------------
        print('predict'.center(50,'-'))
        model.to(device)
        model.eval()
        df_test = df_test.reset_index(drop=True)
        test_dataset = DeepFmDataset(df_test,feature_cfg, is_predict=True) 
        df_result = predict(model, test_dataset, device=device, verbose=True)
        assert df_result.shape[0] == df_test.shape[0]
        df_result['imei'] = df_test['imei']
        df_result['label'] = df_test['label']
        df_result.to_csv(os.path.join(target_dir,'checkpoint_last/eval_prob.csv'),index=False)
        calc_threshold_vs_depth(df_result['label'], df_result['prob'],stats_file=os.path.join(target_dir,'checkpoint_last/eval_result.csv'))
        
        
        ## move to spark 
        import shutil
        shutil.copyfile(all_cfg['tmp']['feat_dict_file'], './spark/features.dict')
        shutil.copyfile('./_config_.yaml', './spark/_config_.yaml')
        shutil.copyfile(os.path.join(target_dir,'checkpoint_last/model.dict'), './spark/model.dict')
        
        
        




    else:   # for predict new data
        # model_path  = os.path.join(base_path, 'server/model/20201202_deepfm_0.7812')
        print('predict')
        all_cfg = yaml.load(open(os.path.join(base_path,'config.yaml'),'r'))
        network_cfg = all_cfg['FM']
        train_cfg = all_cfg['train']
        train_cfg['device'] = device

        pretrained_cfg  = PretrainedConfig(**all_cfg['self_attention']) # for AutoInt
        pretrained_cfg.initializer_range = network_cfg['init_norm_var']
        pretrained_cfg.hidden_size = network_cfg['embedding_size']

        model =  AutoInt(pretrained_cfg, network_cfg, field_size=feature_cfg['field_dim'], feature_size=feature_cfg['feature_dim'])

        # model_path  = '/home/notebook/code/personal/smart_rec/deepFM/server/model/20210111_autoint_lookalike_level2_0.9002/checkpoint_last/model.dict'
        model_path  = '/home/notebook/code/personal/smart_rec/deepFM/server/model/20210111_autoint_lookalike_level2_0.9002/checkpoint_last/model.dict'

        model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))  #map_location=torch.device('cpu')
        model.to(device)
        model.eval()

        df.reset_index(drop=True, inplace=True)
        dataset = DeepFmDataset(df,feature_cfg, is_predict=True)  #is_predict=True  return  label

        df_result = predict(model, dataset, device=device, verbose=True)

        assert df_result.shape[0] == df.shape[0]
        # df_result['imei'] = df['imei'].astype(str)
        df['prob'] = df_result['prob']
        df = df.reindex(columns=['imei','prob'] + feature_cfg['cat_cols'] + feature_cfg['double_cols'])
        df.to_csv('/home/notebook/code/personal/smart_rec/deepFM/server/model/20210111_autoint_lookalike_level2_0.9002/checkpoint_last/lookalike_var_off.csv',index=False)



        
        # print(model)
        # model = torch.load(os.path.join(model_path, 'checkpoint_last/model.pt'), map_location=torch.device(device)) #20201120(64dim)
  

        # df.reset_index(drop=True, inplace=True)
        # dataset = DeepFmDataset(df,feature_cfg, is_predict=False)  #is_predict=True  return  label

        
        
        # # result = evaluate(model, dataset,device=device, verbose=True)
        # # df_result = pd.DataFrame({'label':result['label'],'pred':result['pred']})

        # df_result = predict(model, dataset, device=device,verbose=True)

        # assert df_result.shape[0] == df.shape[0]
        # df_result['imei'] = df['imei'].astype(str)
        # df_result['label'] = df['label']
        # df_result.to_csv('/home/notebook/code/personal/smart_rec/deepFM/server/model/20210107_autoint_lookalike/checkpoint_last/logits.csv',index=False)

        # print('predict done.')
