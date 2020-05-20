import os
import sys
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from fm_preprocessing import DeepFmData, Dataset
from nn_loss_metrics import get_config
from utils import top_ratio_hit_rate, train_sampling, calc_threshold_vs_depth

from deepFM import DeepFM
from xDeepFM import xDeepFM
from AFM import AFM
from utils import colorize
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings(action='ignore')


def train_test_split(Xv,Xi,y, test_ratio=0.1, seed=None):
    index = list(range(len(Xv)))
    Xv = np.asarray(Xv)
    Xi = np.asarray(Xi)
    y = np.asarray(y)
    np.random.seed(seed)
    np.random.shuffle(index)
    test_size = int(len(Xv)*test_ratio)
    test_index = index[-test_size:]
    train_index = index[:-test_size]
    train_Xv = Xv[train_index]
    test_Xv = Xv[test_index]   
    train_Xi = Xi[train_index]
    test_Xi = Xi[test_index]    
    train_y = y[train_index]
    test_y = y[test_index]
    return train_Xv.tolist(), test_Xv.tolist(), train_Xi.tolist(),test_Xi.tolist(), train_y, test_y
    
def data_preprocess(train_data, test_data=None, label='is_y2', deepEnc = None, batch_size=128, 
                    skew_threshold=5, val_ratio=0.2, double_process='z-score', save_h5_file=None, 
                    seed=None):

    train_y = train_data[label].values.reshape(-1,1)
    train_data.drop(columns=[label],inplace=True)
    # ---------------train data
    if deepEnc is None:
        enc = DeepFmData(skew_threshold=skew_threshold,double_process=double_process)
        enc.fit(train_data,y=None)
    else:
        enc = deepEnc

    train_feat_val, train_feat_index = enc.transform(train_data, y=None, normalize_double=True) #list of list
    
    #-----------------val data
    if val_ratio is not None:
        (train_feat_val, val_feat_val, 
        train_feat_index, val_feat_index,
        train_y,val_y ) = train_test_split(train_feat_val, train_feat_index, train_y,test_ratio=val_ratio, seed=seed)
    else:
        (val_feat_val, val_feat_index,val_y) =[None]*3
    
    train_data = Dataset(train_feat_val, train_feat_index, train_y, batch_size, shuffle=True)
    
    #---------------test_data-----------------
    if test_data is not None:
        test_y = test_data[label].values.reshape(-1,1)
        test_data.drop(columns=[label],inplace=True)
        test_feat_val, test_feat_index = enc.transform(test_data, y=None, normalize_double=True)  
        test_data = Dataset(test_feat_val, test_feat_index,test_y, batch_size)
    else:
        (test_feat_val, test_feat_index,test_y) =[None]*3

    if save_h5_file is not None:
        with h5py.File(save_h5_file,'w') as fw:
            train = fw.create_group('train')
            train.create_dataset('train_feat_val', data = np.array(train_feat_val))
            train.create_dataset('train_feat_index',data = np.array(train_feat_index))
            train.create_dataset('train_y', data= np.array(train_y))
            val = fw.create_group('val')
            val.create_dataset('val_feat_val', data =np.array(val_feat_val))
            val.create_dataset('val_feat_index',data= np.array(val_feat_index))
            val.create_dataset('val_y', data=np.array(val_y))
            test = fw.create_group('test')
            test.create_dataset('test_feat_val', data =np.array(test_feat_val))
            test.create_dataset('test_feat_index',data= np.array(test_feat_index))
            test.create_dataset('test_y', data=np.array(test_y))  

    return enc, train_data, test_data, train_feat_val, train_feat_index, train_y, val_feat_val, val_feat_index, val_y

def load_h5_data(h5file, batch_size=128, shuffle=True):
    assert os.path.exists(h5file)
    with h5py.File(h5file, 'r') as fr:
        print('train-null', np.isnan(fr['train']['train_feat_val'].value).sum())
        train_feat_val = fr['train']['train_feat_val'].value.tolist()
        train_feat_index = fr['train']['train_feat_index'].value.tolist()
        train_y = fr['train']['train_y'].value
        train_data = Dataset(train_feat_val, train_feat_index, train_y, batch_size, shuffle=True)

        val_feat_val = fr['val']['val_feat_val'].value.tolist()
        val_feat_index = fr['val']['val_feat_index'].value.tolist()
        val_y = fr['val']['val_y'].value

        test_feat_val = fr['test']['test_feat_val'].value.tolist()
        test_feat_index = fr['test']['test_feat_index'].value.tolist()
        test_y = fr['test']['test_y'].value
        test_data = Dataset(test_feat_val, test_feat_index,test_y, batch_size)
    return train_data, test_data, train_feat_val, train_feat_index, train_y, val_feat_val, val_feat_index, val_y


if __name__ == '__main__':
    import yaml,json
    # pd.set_option('max_colwidth',10)
    # os.environ["CUDA_VISIBLE_DEVICES"] ='0'
    sess_config = get_config(frac=0.4, allow_growth=True, gpu="1")
    pd.set_option('display.max_columns', 20)
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    with open('./conf.yaml','r') as fp:
        config = yaml.load(fp)
    model_type = config['model']
    config = config.get(model_type)
    print(json.dumps(config,indent=2))
    # config = config['deepFM'] if model_type=='deepFM' else config['xDeepFM']
    data_config, params = config['data'],config['params']
    print(' {} '.format(model_type).center(50,'='))
    
    train_file = data_config['train']  #"/home/yuanyuqing163/hb_rl/data/raw/train_bj_dl_200.pkl"
    test_file = data_config['test']  #"/home/yuanyuqing163/hb_rl/data/raw/val_bj_dl_200.pkl"
    train_data = pd.read_pickle(train_file)
    test_data = pd.read_pickle(test_file)
    train_data = train_sampling(train_data, col='is_y2', method='down', pn_ratio=0.2, seed=2020)
    # train_data = train_sampling(train_data, col='is_y2', method='up', pn_ratio=0.5, seed=2019)
    # train_data = train_sampling(train_data, col='is_y2', method='down', pn_ratio=0.5, seed=2019)
    # train_data = train_sampling(train_data, col='is_y2', method='down', pn_ratio=0.05, seed=2019)
    
    if data_config.pop('load_cache'):
        enc = DeepFmData()
        enc.load(data_config['enc_file'])  #'./model/DeepFmData_bjdl200.pkl'
        (train_data, test_data, 
        train_feat_val, train_feat_index, train_y,
        val_feat_val, val_feat_index, val_y) = load_h5_data(data_config['cache_file'], batch_size= params['batch_size'], shuffle=True) #'./data/bj_dl_200.h5'
    else:
        (enc, train_data, test_data, 
        train_feat_val, train_feat_index, train_y,
        val_feat_val, val_feat_index, val_y) = data_preprocess(train_data, test_data,
            deepEnc = None, batch_size= params['batch_size'], skew_threshold=5, val_ratio=0.2, 
            double_process='min-max', save_h5_file=data_config['cache_file'],label='is_y2') 
        enc.save(data_config['enc_file'])

    print(enc._field_dim, enc._feature_dim)
    params.update({'feature_size':enc._feature_dim})
    params.update({'field_size':enc._field_dim})
    
    if model_type.lower()=='deepfm':
        model = DeepFM(params)
    elif model_type.lower() =='xdeepfm':
        model = xDeepFM(params)
    elif model_type.lower() =='afm':
        model = AFM(params)
    else:
        raise ValueError('{} not supported yet'.format(model_type))

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer()) # global_step counter etc.
        sys.stdout.flush()
        best_hit_rate = 0
        best_epoch = 0
        best_loss = np.finfo('float32').max
        stop_cnt = 0
        if params['training_model']:
            #---------------training---------------------------------
            for epoch in range(params['epoch']):
                print('epoch ={}'.format(epoch).center(50,'-'))
                for batch, (xi, xv, y) in enumerate(train_data):
                    # print(xv)
                    step, prob = model.train(sess, xi, xv, y)
                    # print(prob.min(),prob.max())
                    if batch %1000 ==0:
                        train_loss, train_entropy, train_reg = model.evaluate(sess, train_feat_index, train_feat_val, train_y, batch_size=128)
                        print('train_loss=%.4f,\ttrain_ce=%.4f,\treg=%.4f'% (train_loss, train_entropy, train_reg))

                val_loss,val_entropy, val_reg = model.evaluate(sess, val_feat_index, val_feat_val, val_y, batch_size=128)                    
                print('val_loss=%.4f,\tval_ce=%.4f,\treg=%.4f' %(val_loss, val_entropy, val_reg))

                # if epoch%10 ==0 or epoch == params['epoch']-1:
                model.save(sess, model.checkpoint_dir, epoch)
                prob = model.predict(sess, train_feat_index, train_feat_val, batch_size=128)
                hit_rate, top_k = top_ratio_hit_rate(np.array(train_y).ravel(), prob, top_ratio=0.001) # ravel return view, flatten return copy
                train_auc = roc_auc_score(np.array(train_y).ravel(), prob)
                print(colorize('\nk={}, train_1/1000 ={:.4}'.format(top_k ,hit_rate),'cyan',True))

                #-----------------test-----------------------------------
                probs =[]
                ys=[]
                for xi, xv, y in test_data:
                    prob = model.predict(sess, xi, xv)  # list of np.ndarry->array
                    probs.extend(prob.tolist())
                    ys.extend(y.tolist())
                hit_rate, top_k = top_ratio_hit_rate(np.array(ys).ravel(), np.array(probs), top_ratio=0.001)
                val_auc = roc_auc_score(np.array(ys).ravel(), np.array(probs))
                print(colorize('k={}, test_1/1000 ={:.4}'.format(top_k ,hit_rate),'cyan',True))
                print(colorize('train_auc={:.4}, val_auc={:.4}'.format(train_auc,val_auc),'cyan', True))
                if hit_rate > best_hit_rate:
                    best_hit_rate, best_epoch = hit_rate, epoch
                print(colorize('cur_best_rate ={:.4}'.format(best_hit_rate),'cyan',True))
                if hit_rate>0.8:
                    calc_threshold_vs_depth(np.asarray(ys).ravel(), np.asarray(probs))

                # early stopping
                if (val_entropy+5e-5)<best_loss:
                    best_loss = val_entropy
                    stop_cnt = 0
                else:
                    stop_cnt += 1
                    if stop_cnt > 20:
                        break
            print(colorize('epoch={}, best_hit_rate={}'.format(best_epoch, best_hit_rate),'cyan',True))
            
        else:
            model.restore(sess, os.path.split(model.checkpoint_dir)[0])
            probs=[]
            ys =[]
            for xi, xv, y in train_data:
                prob = model.predict(sess, xi, xv)  # np.ndarry
                probs.extend(prob[0].ravel().tolist())
                ys.extend(y.tolist())
            hit_rate, top_k = top_ratio_hit_rate(np.array(ys).ravel(), np.array(probs).ravel(), top_ratio=0.001)
            print('train-top-k={}, train-hit-rate={}'.format(top_k ,hit_rate))
            probs=[]
            ys=[]
            for xi, xv, y in test_data:
                prob = model.predict(sess, xi, xv)  # np.ndarry
                # print(type(prob), prob[0])
                probs.extend(prob[0].ravel().tolist())
                ys.extend(y.tolist())
            hit_rate, top_k = top_ratio_hit_rate(np.array(ys).ravel(), np.array(probs).ravel(), top_ratio=0.001)
            print('test-top-k={}, test-hit-rate={}'.format(top_k ,hit_rate))