import os
import sys
import pandas as pd 
import numpy as np
import tensorflow as tf  
from losses import focal_loss, weighted_binary_crossentropy
from utils import Dataset

class DeepFM(object):
    def __init__(self,params):
        self.feature_size=params['feature_size']
        self.field_size = params['field_size']
        self.emb_size = params['emb_size']
        self.deep_layers = params['deep_layers']
        self.l2_reg_coef = params['l2_reg_coef']
        self.lr = params['lr']
        self.pos_ratio = params['pos_ratio']
        self.keep_prob_v = params['keep_prob_v']
        self.activate = params['activate']
        self.weight={}
        self.saver=None 
        self.checkpoint_dir = params['checkpoint_dir']
        self.build()

    def _init_placeholders(self):
        self.feat_index =tf.placeholder(tf.int32, shape=[None,None],name='feat_index')
        self.feat_value = tf.placeholder(tf.float32, shape=[None,None],name='feat_value')
        self.label = tf.placeholder(tf.float32, shape=[None, 1],name='label')
        self.keep_prob = tf.placeholder(tf.float32,shape=[],name='keep_prob')
        self.is_training = tf.placeholder(tf.bool, shape=[],name='is_training')

    def build(self):
        """
        feature_size:N 
        field_size:F 
        emb_size: K 
        batch_size:None 
        """
        self._init_placeholders()

        #1. ----------------------定义权值---------------------------
        #FM 一次项权值
        self.weight['first_order']= tf.Variable(tf.random_normal([self.feature_size,1],0.0,0.05),name='first_order')
        #one-hot编码后的输入层与Dense Embedding 层的权值定义，DNN的输入Embedding
        self.weight['emb_weight'] = tf.Variable(tf.random_normal([self.feature_size,self.emb_size],0.0,0.05),name='emb_weight')

        #deep部分的weight和bias， deep网络初始输入维度：input_size = F*K
        layers = len(self.deep_layers)
        input_size = self.field_size * self.emb_size
        # glorot= np.sqrt(2.0/(input_size+self.deep_layers[0])) # sigmoid
        he_normal = np.sqrt(2.0/input_size)
        self.weight['layer_0'] = tf.Variable(np.random.normal(loc=0,scale=he_normal,size=(input_size, self.deep_layers[0])),dtype=np.float32)
        self.weight['bias_0'] = tf.Variable(np.random.normal(loc=0,scale=he_normal,size=(1,self.deep_layers[0])),dtype=np.float32)
        #生成deepnet里面的weight和bias
        for i in range(1,layers):
            he_normal = np.sqrt(2.0/self.deep_layers[i-1])
            self.weight['layer_'+str(i)] = tf.Variable(np.random.normal(loc=0, scale=he_normal,size=(self.deep_layers[i-1],self.deep_layers[i])),dtype=np.float32)
            self.weight['bais_'+str(i)] = tf.Variable(np.random.normal(loc=0, scale=he_normal, size=(1, self.deep_layers[i])),dtype=np.float32)
        
        # deep部分output_size + 一次项output_size+ 二次项output_size
        last_layer_size = self.deep_layers[-1] + self.field_size + self.emb_size
        glorot_normal = np.sqrt(2.0/(last_layer_size+1))
        #最后一次的weight+bias
        self.weight['last_layer'] = tf.Variable(np.random.normal(loc=0, scale=glorot_normal, size=(last_layer_size,1)),dtype=np.float32)
        self.weight['last_bias'] = tf.Variable(tf.constanct(0.0),dtype=np.float32)

        #2.-----------------------------前向传播---------------------------------------
        self.emb_index = tf.nn.embedding_lookup(self.weight['emb_weight'],self.feat_index)  #None*F*K
        self.emb_part = tf.multiply(self.emb_index, tf.reshape(self.feat_value, [-1, self.field_size,1])) #[None*F*K].*[None*F*1] = [None*F*K]

        # FM 一阶特征None*F
        self.emb_first = tf.nn.embedding_lookup(self.weight['first_order'],self.feat_index)  #None*F*1
        self.emb_first = tf.multiply(self.emb_first,tf.reshape(self.feat_value, [-1, self.field_size,1]))
        self.first_order = tf.reduce_sum(self.emb_first,2) #None*F 

        #FM 二阶特征None*K
        self.sum_second_order = tf.reduce_sum(self.emb_part, 1) #None*K
        self.sum_second_order_square = tf.square(self.sum_second_order)
        self.square_second_order = tf.square(self.emb_part) 
        self.square_second_order_sum = tf.reduce_sum(self.square_second_order,1) #None*K 
        #ab = 1/2*[(a+b)*2-a^2-b^2]
        self.second_order = 0.5 * tf.subtract(self.sum_second_order_square- self.square_second_order_sum) #None*K

        #FM 输出None*(F+K)
        self.fm_part = tf.concat([self.first_order,self.second_order],axis=1)

        #DNN 部分
        self.deep_emb = tf.reshape(self.emb_part, [-1, self.field_size*self.emb_size]) # None*(F*K)
        for i in range(0,layers):
            self.deep_emb = tf.add(tf.matmul(self.deep_emb,self.weight['layer_'+str(i)]),self.weight['bias_'+str(i)])
            self.bn_out = tf.layers.batch_normalization(self.deep_emb, training = self.is_training)
            self.deep_emb = self.activate(self.deep_emb)
            # self.deep_emb = tf.layers.dropout(sefl.deep_emb, rate =1.0- self.keep_prob, training = self.is_training)
        
        #FM 部分和DNN拼接
        din_all = tf.concat([self.fm_part, self.deep_emb],axis=1)
        self.out = tf.add(tf.matmul(din_all, self.weight['last_layer'])+self.weight['last_bias'])

        #3----------------------------确定损失------------------------
        self.prob = tf.nn.sigmoid(self.out)
        # self.entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.label, logits=sefl.out)
        # self.entropy_loss = - tf.reduce_mean(self.label*tf.log(tf.clip_by_value(self.prob, 1e-10,1.0))+(1.0-self.label)*tf.log(tf.clip_by_value(1.0-self.prob,1e-10,1.0)))
        self.entropy_loss = focal_loss(self.prob, self.label,alpha=0.5, gamma=2)

        self.reg_loss = tf.contrib.layers.l2_regularizer(self.l2_reg_coef)(self.weight['last_layer'])
        for i in range(layers):
            self.reg_loss += tf.contrib.layers.l2_regularizer(self.l2_reg_coef)(self.weight['layer_'+str(i)])
        #print(self.reg_loss.shape.as_list())
        self.loss = self.entropy_loss + self.reg_loss
        
        self.global_step = tf.Varaible(0, trainable=False, name='global_step')
        self.lr = tf.train.exponential_decay(self.lr, self.global_step, 3000, 0.99, staricase=Fasle)
        optimizer =tf.train.AdamOptimizer(self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        trainable_vars = tf.trainable_variables()
        gradients  tf.gradients(self.loss, trainable_vars)
        clip_gradients,_ = tf.clip_by_global_norm(gradients,5)
        with tf.control_dependency(update_ops):
            self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainble_vars),global_step = self.global_step)
        self.saver = tf.train.Saver(max_to_keep=3)

    def train(self,sess, feat_index, feat_value, label):
        _, step = sess.run([self.train_op, self.global_step],feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value,
            self.label:label,
            self.keep_prob: self.keep_prob_v,
            self.is_training:True    
        })
        return step
    
    def predict(self,sess, feat_index, feat_value, batch_size=None):
        if batch_size is None:
            prob = sess.run(self.prob, feed_dict={self.feat_index:feat_index, self.feat_value:feat_value,self.keep_prob:1,
            self.is_training:False})
        else:
            data = Dataset(feat_value, feat_index,[None]*len(feat_index),batch_size,shuffle=False)
            probs = []
            for feat_index, feat_value, _ in data:
                prob = sess.run([self.prob], feed_dict={self.feat_index:feat_index, self.feat_value:feat_value,self.keep_prob:1,
                                        self.is_training:False})[0]
                probs.append(prob.ravel())
            prob = np.concatenate(probs)
        return prob.ravel()

    def evaluate(self, sess, feat_index,feat_value, label, batch_size=None ):
        tloss, entloss, regloss = 0,0,0
        if batch_size is None:
            tloss, entloss, regloss = sess.run([self.loss, self.entropy_loss, self.regloss],feed_dict={
                self.feat_index: feat_index,
                self.feat_value: feat_value,
                self.label:label,
                self.keep_prob: self.keep_prob_v,
                self.is_training:False    
            })
        else:
            data = Dataset(feat_value,feat_index, label,batch_size, shuffle=Fasle)
            for i,(feat_index,feat_value,label) in enumerate(data,1):
                _tloss, _entloss, _regloss = sess.run([self.loss, self.entropy_loss, self.reg_loss],feed_dict={
                    self.feat_index: feat_index,
                    self.feat_value: feat_value,
                    self.label:label,
                    self.keep_prob: self.keep_prob_v,
                    self.is_training:False   
                })
                tloss = tloss +(_tloss - tloss)/i
                entloss = entloss + (_entloss- entloss)/i 
                regloss = regloss + (_regloss - regloss)/i 

        return tloss, entloss, regloss 
     
    def save(self, sess, path, global_step):
        if self.saver is not None:
            self.saver.save(sess, save_path=path, global_step=global_step)

    def restore(self,sess, path):
        model_file = tf.train.latest_checkpoint(path)
        if model_file is not None:
            print('restore_model:', model_file)
            self.saver.restore(sess,save_path=model_file)








      















        #



