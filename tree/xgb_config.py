import os
import datetime

def load_xgb_config(root_path, data_path, docs_path, model_path, tmp_path, log_path,
                    version,model_id, bussiness_type='zj'):
    print('load_xgb_config'.center(50,'-'))
    
    # root_path = "/home/yuanyuqing163/hb_model/" 
    
    feature_config={'iv_threshold': 1e-5,#001
                    'xgb_importance_revise': True,   #False->use exist importance to train model
                    'xgb_importance_type': 'gain',  # gain, weight
                    'imp_threshold':11e-4,      # 0.0008, 0.00067 
                    'null_percent':0.98,
                    'category_threshold': 50,
                    'drop_less': False,
                    'corr_threshold': 0.99,
                    }
        
    xgb_config ={ 
                'objective': 'binary:logistic',
                'booster': 'gbtree', 'silent': True,
                'eval_metric': ['logloss', 'aucpr', 'auc'], 
                'eta': 0.03535927569387097, 
                'max_depth': 6, 
                'subsample': 0.7978975772673077, 
                'colsample_bytree': 0.9908627391242886, 
                'colsample_bylevel': 1, 
                'min_child_weight': 3.7814985568573136, 
                'min_split_loss': 1.3784784955327494, 
                'scale_pos_weight': 1, 
                'lambda': 2.8969397592269037, 
                'alpha': 0.4066102366819282, 
                'seed': 0,
                'tree_method': 'hist', 
                'grow_policy': 'lossguide',
                'max_leaves': 28, 
                'max_bin': 255, 
                "nthread": 6,
                "num_round": 500,
                "early_stopping_rounds": 40,
                'feval':None,
                'maximize':True,
                "verbosity": 1
                }
    ##=========================  file ===========================
    pjoin = os.path.join
    data_path = pjoin(root_path, data_path,'trans') 
    conf_path = pjoin(root_path,docs_path)
    tmp_path = pjoin(root_path,tmp_path,bussiness_type)
    # model_path = pjoin(root_path,model_path,'xgb',bussiness_type+datetime.datetime.now().strftime('_%m%d'))
    model_path = pjoin(root_path,model_path,version,'xgb',model_id+'_'+bussiness_type+'_model')
    log_path = pjoin(root_path,log_path)
    if not os.path.exists(data_path): os.makedirs(data_path)
    if not os.path.exists(tmp_path): os.makedirs(tmp_path)
    if not os.path.exists(model_path): os.makedirs(model_path)
    if not os.path.exists(log_path): os.makedirs(log_path)

    input_file=dict(
                    iv_file =  None,
                    imp_file = pjoin(model_path,'xgb_importance_'+ feature_config['xgb_importance_type']+'.csv'),
                    fmap = pjoin(model_path,'xgb.fmap'),
                    cv_train_file =  pjoin(data_path,'train_cv_xgb_'+model_id+'_'+bussiness_type),
                    cv_val_file = pjoin(data_path, 'val_cv_xgb_'+model_id+'_'+bussiness_type),
                    )
                    
    output_file = dict(model_path = pjoin(model_path,'xgb.model'),
                       val_attr_path = pjoin(model_path,'xgb_val_attr'),
                       null_path = pjoin(tmp_path,'drop_null.csv'),
                       category_path = pjoin(tmp_path,'drop_category.csv'),
                       low_iv_path = pjoin(tmp_path,'drop_low_iv.csv'),
                       corr_path = pjoin(tmp_path,'drop_corr.csv'),
                       encoder_path = pjoin(model_path,'xgb_encoder'),
                       tmp_path = tmp_path,
                       select_feature_path = pjoin(model_path,'xgb_selected_feature.csv'),
                       log_file = pjoin(log_path, 'xgb_log'+ datetime.datetime.now().strftime('_%m%d')))

    return feature_config, xgb_config, input_file, output_file
