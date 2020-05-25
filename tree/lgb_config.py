import os
import datetime
from math import floor,exp
import numpy as np
# from .utils import feval_top_hit_lgb, feval_aucpr_lgb


# def binary_loss(preds, train_data))
    # labels = train_data.get_label()
    # preds = 1. / (1. + np.exp(-preds))
    # grad = preds - labels
    # hess = preds * (1. - preds)
    # return grad, hess
def feval_aucpr_lgb(y_pred, dtrain):
    y_true = dtrain.get_label()
    precisions, recalls ,thrs = precision_recall_curve(y_true, y_pred)
    mean_precisions = 0.5*(precisions[:-1]+precisions[1:])
    intervals = recalls[:-1] - recalls[1:]
    auc_pr = np.dot(mean_precisions, intervals)
    return 'aucpr', auc_pr, True 
    
alpha=0.7
def lgb_weight_binary_loss(preds, train_data):  # lightgbm ->log(1+exp(-yaF))
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = (alpha*labels+(1-alpha)*(1-labels))*preds - alpha*labels
    hess = (alpha*labels+(1-alpha)*(1-labels))*preds*(1-preds)
    return grad, hess

gamma = 2
def focal_loss(preds, train_data):  # output will be LLR, not probability
    yt = train_data.get_label() # series
    yt = yt.values 
    yp = 1./(1. + np.exp(-preds))
    g1 = yp*(1-yp)
    g2 = yt + np.power(-1,yt)*yp
    g3 = yp + yt -1
    g4 = 1-yt-np.power(-1,yt)*yp
    g5 = yt +np.power(-1,yt)*yp
    grad = gamma*g3*np.power(g2, gamma)*np.log(g4)+np.power(-1,yt)*np.power(g5, gamma+1)
    # grad = gamma*(yp + yt-1)*(yt+(-1)^(yt)*yp)^gamma*np.log(1-yt-(-1)^yt*yp) +(-1)^yt*(yt+(-1)^yt*yp)^(gamma+1)
    hess = g1*(gamma*((np.power(g2, gamma)+ gamma*np.power(-1,yt)*g3*np.power(g2,gamma-1))*np.log(g4)-(np.power(-1,yt)*g3*np.power(g2,gamma))/g4)+(gamma+1)*np.power(g5,gamma))
    return grad, hess

def load_lgb_config(root_path, data_path, docs_path, model_path, tmp_path, log_path, version,model_id,
                    bussiness_type,imp_file):
    print('load_lgb_config'.center(50,'-'))
    # root_path = r'/home/yuanyuqing163/hb_model'

    feature_config={'iv_threshold': 0.00001,           # 0.002
                    'lgb_importance_revise': False, # False->use exist importance to train model
                    'lgb_importance_type': 'gain',  # gain, split
                    'imp_threshold': 13.5e-4,       # 15(hnb,125),18(hnb2,126),21(hnb2,104)
                    'category_threshold': 50,
                    'null_percent':0.98,
                    'drop_less':False,   # True, keep all attrs with importance above threhold
                    'corr_threshold':0.98,  #0.95(142_xgbimp) 0.92(13-159)
                    'model_type':'lgb',
                    }

    lgb_config={
                "learning_rate": 0.022647037734232003,
                "max_depth": 7,
                "num_leaves": 49,
                "min_data_in_leaf": 39,
                "bagging_fraction": 0.7863746352956377,
                "bagging_freq": 1,
                "feature_fraction": 0.827604087681333,
                "min_gain_to_split": 1.8335341943609902,
                "min_data_in_bin": 22,
                "lambda_l2": 2.2635889734158456,
                "lambda_l1": 0.2791470419628548,
                "seed": 42,
                "num_threads": 8,
                "num_boost_round": 800, 
                "early_stopping_round": 40,
                "min_sum_hessian_in_leaf": 0.001,
                "max_cat_threshold": 16,   # 32, limit the max threshold points in categorical features

                'fobj': None,
                "feval": None,
                # "learning_rates": lambda x: 0.002 if x<5 else 0.03*exp(-floor(x/200)),#[0,n_iters)
                "learning_rates": None,
                "objective": "binary",
                "is_unbalance": False,   # or scale_pos_weight = 1.0
                'zero_as_missing':False,
                "metric": ["binary_logloss","auc"],
                "metric_freq": 5,
                "boosting": "gbdt",
                "verbose": 0,   #<0(fatal) 0(waring) 1(info) >1(debug)
                'boost_from_average':True  # default True
              }

    ##=========================file ===========================
    
    data_path = os.path.join(root_path, data_path,'trans') 
    conf_path = os.path.join(root_path,docs_path)
    tmp_path = os.path.join(root_path,tmp_path,bussiness_type)
    # model_path = os.path.join(root_path,model_path,'lgb',bussiness_type+ datetime.datetime.now().strftime('_%m%d'))
    model_path = os.path.join(root_path,model_path,version, 'lgb',model_id+'_'+bussiness_type+ '_model')
    log_path = os.path.join(root_path,log_path)
    if not os.path.exists(data_path): os.makedirs(data_path)
    if not os.path.exists(tmp_path): os.makedirs(tmp_path)
    if not os.path.exists(model_path):os.makedirs(model_path)
    if not os.path.exists(log_path):os.makedirs(log_path)

    input_file=dict(
                    # iv_file =  os.path.join(conf_path,'iv_bj_lx.csv'),
                    iv_file =  None,
                    lgb_imp_file = os.path.join(model_path,'lgb_importance_total.csv'),
                    lgb_imp_tmp_file = os.path.join(model_path,'lgb_importance_cur.csv'),
                    imp_file = imp_file,
                    cv_train_file =  os.path.join(data_path,'train_cv_lgb_'+model_id+'_'+bussiness_type),
                    cv_val_file = os.path.join(data_path, 'val_cv_lgb_'+model_id+'_'+bussiness_type),
                    category_feature_path = os.path.join(model_path, 'lgb_category_feat'),
                    )

    output_file = dict(model_path = os.path.join(model_path,'lgb.model'),
                       # val_attr_path = os.path.join(model_path,'lgb_val_attr'),
                       null_path = os.path.join(tmp_path,'drop_null.csv'),
                       category_path = os.path.join(tmp_path,'drop_category.csv'),
                       encoder_path = os.path.join(model_path,'lgb_encoder'),
                       low_iv_path = os.path.join(tmp_path,'drop_low_iv.csv'),
                       corr_path = os.path.join(tmp_path,'drop_corr.csv'),
                       select_feature_path = os.path.join(model_path,'lgb_selected_feature.csv'),
                       log_file = os.path.join(log_path, 'lgb_log'+ datetime.datetime.now().strftime('_%m%d')))

    return feature_config, lgb_config, input_file, output_file    

