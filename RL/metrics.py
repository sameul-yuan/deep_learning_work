# -*- encoding:utf-8 -*-
import os 
import warnings 
import numpy as np 
import tensorflow as tf 
from sklearn.metrics import roc_auc_score 
# from tensorflow.python.keras.callbacks import Callback
# from tensorflow.python.keras import backend as K 
from tensorflow import keras 
K = keras.backend

def tf_summary(tags:list,vals:list):
    """
    scalar value TensorFlow Summary
    """
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag,value in zip(tags,vals)])

def auroc(y_true, y_pred): # for metric
    return tf.py_func(roc_auc_score, (y_true,y_pred),tf.double)

class AucROC(keras.callbacks.Callback):
    def __init__(self, train_data, val_data):
        super(AucROC,self).__init__()
        self.x,self.y = (train_data[0],train_data[1])
        self.val_x, self.val_y =(val_data[0],val_data[1])

    def on_train_begin(self,logs={}):
        return 
    def on_train_end(self, logs={}):
        return 
    def on_epoch_begin(self, logs={}):
        return  
    
    def on_epoch_end(self,logs={}):
        logs['val_auc'] = float('-inf')
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred.reshape(-1))

        val_y_pred = self.model.predict(self.val_x)
        val_roc = roc_auc_score(self.val_y, val_y_pred.reshape(-1))
        logs['val_auc'] = roc_val 
        print('train_auc:%s val-auc:%s'%(str(round(roc,4),str(round(val_auc,4)))))
     

class WeightedBinaryCrossEntropy(object):
    def __init__(self, pos_ratio, from_logits=False):
        self.weights = tf.constant((1.0-pos_ratio)/pos_ratio,tf.float32)
        self.pos_ratio = tf.constant(pos_ratio,tf.float32)
        self.from_logits = from_logits
        self.__name__ = "weighted_binary_cross_entropy_{}".format(pos_ratio)
    
    def __call__(self, y_true,y_pred):
        return self.weighted_binaray_cross_entorpy(y_true,y_pred)

    def weighted_binaray_cross_entorpy(self,y_true,y_pred):
        epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
        if not self.from_logits:
            y_pred= tf.clip_by_value(y_pred, epsilon, 1-epsilon)
            y_pred = tf.log(y_pred/(1-y_pred))
        cost = tf.nn.weighted_cross_entropy_with_logits(y_true,y_pred,self.weights)
        return K.mean(cost*self.pos_ration,axis=-1)

def focal_loss(gamma=2, alpha=0.25):
    def binary_focal_loss(y_true,y_pred):
        pt_1 = tf.where(tf.equal(y_true,1),y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true,0),y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha*K.pow(1.-pt_1,gamma)*K.log(K.epsilon()+pt_1)) - K.sum((1.-alpha)*K.pow(pt_0,gamma)*K.log(1.0-pt_0+K.epsilon()))
    return binary_focal_loss


