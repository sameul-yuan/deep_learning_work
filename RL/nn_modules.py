import os 
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 

def get_session_config(frac=0.4, allow_growth=True, gpu="0"):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu 
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    config = tf.ConfigProto()
    if tf.test.is_gpu_available():
        print("gpu:{}".format(gpu))
        config.gpu_options.per_process_gpu_memory_fraction = frac 
        config.gpu_options.allow_growth = allow_growth
    config.log_device_placement=False 
    config.allow_soft_placement=True 
    return config

def dense_batchnorm_layer(x,units, activation='relu', kernel_initializer='he_normal', kernel_regularizer=None, name=None, 
                            batch_norm=True):
    x = keras.layers.Dense(units=units, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,name=name)(x)
    if batch_norm:
        x = keras.layers.BatchNormalization()(x)
    if activation == 'lrelu':
        x = keras.layers.LeakyReLU(alpha=0.3)(x)
    elif activation=='prelue':
        x = keras.layers.PReLU()(x)
    else:
        x = keras.layers.Activation(activation)(x)
    return x
    
def dense_dropout_layer(x,unit,activation='relu',kernel_initializer='he_normal',kernel_regularizer=None, drop_rate=0.5,
                        training=None,name=None):
    x = keras.layers.Dense(units=unit, activation=activation, kernel_initializer=kernel_initializer, 
                            kernel_regularizer=kernel_regularizer,name=name)(x)
    x = keras.layers.Dropout(rate=drop_rate)(x, training=training)
    return x

def dense_drop_norm_layer(x,unit, activation='relu',kernel_initializer='he_normal',kernel_regularizer=None, 
                        batch_norm=True, drop_rate=0.5, training=None, name=None):
    x = keras.layers.Dense(units=unit, kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer,
                    name=name)(x)
    if batch_norm:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)
    x = keras.layers.Dropout(rate=drop_rate)(x, training=training)
    return x

def label_smoothing(y: np.ndarray, epsilon=0.1):
    classes = 2 if y.ndim==1 else y.shape[-1]
    return (1.0-epsilon)*y + epsilon/classes