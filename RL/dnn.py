# -*- coding: utf-8 -*-
"""
"""
import sys
import os
import warnings
import pandas as pd
from datetime import datetime

from sklearn.model_selection import  train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.metrics import  binary_accuracy
from tensorflow.python.keras.layers import Input, Dense, Activation,Dropout,BatchNormalization,Flatten
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.python.keras import regularizers
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.activations import relu
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard,ReduceLROnPlateau
import tensorflow.keras.backend as K


# sys.path.append(os.path.dirname(os.path.split(os.path.abspath(__file__))[0]))
from metrics import (RocAuc, auroc, get_session_config, WeightedBinaryCrossEntropy,focal_loss)
# from utils.nn_utils import  train_sampling
from logger import colorize
from nn_modules import dense_batchnorm_layer,dense_dropout_layer

# tf.compat.v1.disable_eager_execution()

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',4 )

def load_data(train_pkl, test_pkl, label='is_y2', method=None, target_pn_ratio=None,seed=None):
    train_data = pd.read_pickle(train_pkl)
    test_data = pd.read_pickle(test_pkl)
    print(colorize('train-shape={}\t test_shape={}'.format(train_data.shape, test_data.shape), 'blue', True))
    print(train_data.head())
    print(test_data.head())

    pn_ratio = sum(train_data.is_y2 == 1) / sum(train_data.is_y2 == 0)
    print(colorize('naive-pn-ratio={:.4f}'.format(pn_ratio), 'blue', True))

    if target_pn_ratio:
        method = method or 'up'
        train_data = train_sampling(train_data, col=label, method=method, pn_ratio=target_pn_ratio,seed=seed)
    train_data = shuffle(train_data, random_state=42)

    print('shuffle:\n', train_data.head(20))
    print(colorize('train-shape={}'.format(train_data.shape),'blue',True))

    train_y = train_data[label]
    train_x = train_data.drop(columns=[label])
    test_y = test_data[label]
    test_x = test_data.drop(columns=[label])
    assert train_x.isna().sum().sum() == 0
    return train_x, train_y, test_x, test_y

class DNN(object):

    def __init__(self,input_dim, out_dim):
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.model = self.build()
        self.callbacks = []
        self._loss = None
        self._opt = None
        self._metrics = None
        self._compile=False


    def build(self):
        inputs = keras.Input(shape=(self.input_dim,),name='in')
        x = dense_batchnorm_layer(inputs, 64, activation='relu', kernel_initializer='he_normal',
                                  kernel_regularizer=None, batch_norm=False, name='fc1')
        x = dense_batchnorm_layer(x,64, activation='relu',kernel_initializer='he_normal',
                                  kernel_regularizer=None, name='fc2')
        x = dense_batchnorm_layer(x,64, activation='relu',kernel_initializer='he_normal',
                                  kernel_regularizer=None, name='fc3')
        x = keras.layers.Dropout(0.5)(x)
        outputs = dense_batchnorm_layer(x,self.out_dim, activation='sigmoid',batch_norm=False)

        return keras.Model(inputs=inputs, outputs=outputs)

    def add_naive_callbacks(self):

        self.callbacks.append(EarlyStopping(monitor='val_loss', patience=20, mode='min',
                                       min_delta=0.0002))  # monitor = quanlity in callback 'logs'
        self.callbacks.append(ModelCheckpoint("../model/dnn/model_rate_{test_hit_rate:.3f}-" + datetime.now().strftime('%m%d') + \
                                      ".model",monitor='test_hit_rate', save_best_only=True, mode='max', period=10))
        # self.callbacks.append(TensorBoard(log_dir='../assets', histogram_freq=1, write_grads=True, write_images=True))
        self.callbacks.append(ReduceLROnPlateau('val_loss', factor=0.5, patience=20, mode='min', min_lr=1e-4))

    def compile(self):
        assert self.loss is not None and self.optimizer is not None
        self.model.compile(loss=self.loss,  optimizer=self.optimizer,metrics= self.metrics)
        self._compile = True

    def fit(self,train_x, train_y, batch_size=64, epochs=2000, verbose=1, shuffle=True,validation_data=None,
            class_weight=None, callbacks=None, **kwargs): # class_weight={0:1,1:20}
        if not self._compile:
            self.compile()
        callbacks  = self.callbacks or callbacks
        history = self.model.fit(train_x, train_y,batch_size = batch_size,epochs= epochs, verbose=verbose, shuffle= shuffle,
                       validation_data =validation_data, class_weight=class_weight, callbacks=callbacks, **kwargs)
        return history

    @property
    def loss(self):
        return self._loss
    @loss.setter
    def loss(self, loss):
        self._loss = loss

    @property
    def optimizer(self):
        return self._opt
    @optimizer.setter
    def optimizer(self,opt):
        self._opt = opt

    @property
    def metrics(self):
        return self._metrics
    @metrics.setter
    def metrics(self,metrics):
        self._metrics = metrics

if  __name__ =='__main__':
    #------------------load data ----------------------------
    root_data = r'/home/yuanyuqing163/hb_rl/data/clean/'
    train_file = os.path.join(root_data,'train_bj_dl_107_woe_tail.pkl')
    test_file = os.path.join(root_data,'val_bj_dl_107_woe_tail.pkl')
    train_x, train_y, test_x, test_y = load_data(train_file, test_file,method=None, target_pn_ratio=None,seed=None)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, stratify=train_y, test_size=0.1,random_state=42)
    pos_ratio = sum(train_y == 1) / len(train_y)

    keras.backend.set_session(get_session(frac=0.4, allow_growth=True, gpu="1"))
    net = DNN(input_dim=train_x.shape[1], out_dim=1)
    #callbacks
    net.callbacks.append(TopRatioHitRate([test_x.values,test_y.values]))
    # net.callbacks.append(RocAuc([train_x.values,train_y.values],[val_x.values,val_y.values]))
    net.add_naive_callbacks()

    #compile & fit
    net.loss = focal_loss(gamma=2, alpha=0.9)   # WeightedBinaryCrossEntropy(pos_ratio), focal_loss(gamma=2, alpha=0.9)
    net.optimizer = Adam(lr=0.001)  # Adam(lr=0.001)  SGD(lr=0.001,nesterov=False)
    history = net.fit(train_x.values, train_y.values, batch_size=64, epochs=200, verbose=1, shuffle=True,
                      validation_data=(val_x.values, val_y.values),class_weight=None)

    df = pd.DataFrame(history.history)
    df.to_csv('../assets/misc_'+datetime.now().strftime('%m%d_%H%M')+'.csv',index=False)
    print(colorize('done'.center(50,'-'),'green',True))



# model = Sequential()
# l1 =0
# l2 =0
# model.add(Dense(units=64, activation='relu',input_dim=train_x.shape[1],
#                 kernel_regularizer=None,            #regularizers.l1_l2(l1=l1, l2=l2)
#                 kernel_initializer = 'he_normal',name='fc1'))
#
# model.add(Dense(units=64, activation=None, kernel_regularizer=None,
#                 kernel_initializer='he_normal',name='fc2'))
# model.add(BatchNormalization())
# model.add(Activation(activation='relu'))
#
# model.add(Dense(units=64, activation=None, kernel_regularizer=None,
#                 kernel_initializer='he_normal'))
# model.add(BatchNormalization())
# model.add(Activation(activation='relu'))
# model.add(Dropout(0.5))
#
#
# model.add(Dense(units=1, activation='sigmoid',name='out'))
#
# # keras.utils.plot_model(model, '../assets/dnn.png', True, True)
#
# optimizer = SGD(lr=0.001,nesterov=False)
# # optimizer = Adam(lr=0.001)
# model.compile(loss=focal_loss(gamma=2, alpha=0.9),   #binary_crossentropy  WeightedBinaryCrossEntropy(pos_ratio), focal_loss(gamma=2, alpha=0.9)
#               optimizer=optimizer, #'sgd'
#               metrics=None)   #[binary_accuracy]
#
# early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min',min_delta=0.0002)  # monitor = quanlity in callback 'logs'
# model_check = ModelCheckpoint("../model/dnn/model_rate_{test_hit_rate:.3f}-"+datetime.now().strftime('%m%d')+".model",
#                               monitor='test_hit_rate',save_best_only=True,mode='max',period=10)
# tensorbord = TensorBoard(log_dir='../assets', histogram_freq=1,write_grads=True, write_images=True)
# schedule_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=20, mode='min', min_lr=1e-4)
# hit_rate=TopRatioHitRate([test_x.values,test_y.values])
# auc= RocAuc([train_x.values,train_y.values],[val_x.values,val_y.values])
# cbks = [hit_rate, model_check,early_stopping,schedule_lr]
#
# history = model.fit(train_x.values, train_y.values, batch_size=64, epochs=2000, verbose=1, shuffle=True,
#                     validation_data=(val_x.values, val_y.values),
#                     class_weight=None,  #{0:1,1:20}
#                     callbacks=cbks)
#
# df = pd.DataFrame(history.history)
# df.to_csv('../assets/misc_'+datetime.now().strftime('%m%d_%H%M')+'.csv',index=False)

# loss_and_metric= model.evaluate(X,y)  #list
##functional API
#inputs = Input(shape=(784,)) # no batch size
#x= Dense(64, activation='relu')(inputs)
#x= Dense(64, activation='relu')(x)
#pred = Dense(1, activation='sigmoid')(x)
#model = Model(inputs=inputs, outputs= pred)
#model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])
#model.fit(X,y,batch_size=32)
#
## reuse layers and model
#input1 = Input(784)
#input2 = Input(784)
#dense = Dense(64,input_dim=10)
#output1 = dense(input1)
#output2 = dense(input2)
#y_1= model(input1)
#y_2= model(input2)