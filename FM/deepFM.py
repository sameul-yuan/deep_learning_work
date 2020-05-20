import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from losses import focal_loss,weighted_binary_crossentropy
from utils import Dataset

class DeepFM(object):
    def __init__(self, params):
        self.feature_size = params['feature_size']
        self.field_size = params['field_size']
        self.embedding_size = params['embedding_size']
        self.deep_layers = params['deep_layers']
        self.l2_reg_coef = params['l2_reg']
        self.learning_rate = params['learning_rate']
        self.pos_ratio = params['pos_ratio']
        self.keep_prob_v = params['keep_prob']
        self.activate = tf.nn.relu
        self.weight = {}
        self.saver=None
        self.checkpoint_dir = params['checkpoint_dir']
        self.build()

    def build(self):
        """
        feature_size: N
        field_size: F
        embedding_size: K
        batch_size:  None
        """
        self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feature_index')
        self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feature_value')
        self.label = tf.placeholder(tf.float32, shape=[None,1], name='label')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob') # scaler
        self.is_training= tf.placeholder(tf.bool, shape=[],name='is_training')
        
        #1、-------------------------定义权值-----------------------------------------
        # FM部分中一次项的权值定义
        self.weight['first_order'] = tf.Variable(tf.random_normal([self.feature_size, 1], 0.0, 0.05), # N * 1
                                                    name='first_order')
        # One-hot编码后的输入层与Dense embeddings层的权值定义，即DNN的输入embedding。
        self.weight['embedding_weight'] = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.05),   # N*K
                                                    name='embedding_weight')
        # deep网络部分的weight和bias, deep网络初始输入维度：input_size = F*K
        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        # glorot_normal = np.sqrt(2.0 / (input_size + self.deep_layers[0])) #　for sigmoid 
        he_normal = np.sqrt(2.0 /input_size)   # for relu

        self.weight['layer_0'] = tf.Variable(np.random.normal(loc=0, scale=he_normal, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        self.weight['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=he_normal, size=(1, self.deep_layers[0])), dtype=np.float32)

        # 生成deep network里面每层的weight 和 bias
        for i in range(1, num_layer):
            he_normal = np.sqrt(2.0 / (self.deep_layers[i - 1]))
            self.weight['layer_' + str(i)] = tf.Variable(np.random.normal(loc=0, scale=he_normal, size=(self.deep_layers[i - 1], self.deep_layers[i])),
                                                       dtype=np.float32)
            self.weight['bias_' + str(i)] = tf.Variable(np.random.normal(loc=0, scale=he_normal, size=(1, self.deep_layers[i])),dtype=np.float32)

        # deep部分output_size + 一次项output_size + 二次项output_size 
        last_layer_size = self.deep_layers[-1] + self.field_size + self.embedding_size
        glorot_normal = np.sqrt(2.0 / (last_layer_size + 1))
        # 生成最后一层的weight和bias
        self.weight['last_layer'] = tf.Variable(np.random.normal(loc=0, scale=glorot_normal, size=(last_layer_size, 1)), dtype=np.float32)
        self.weight['last_bias'] = tf.Variable(tf.constant(0.0), dtype=np.float32)

        #2、----------------------前向传播------------------------------------
        # None*F*K
        self.embedding_index = tf.nn.embedding_lookup(self.weight['embedding_weight'],self.feat_index)  
        # [None*F*K] .*[None*F*1] = None*F*K
        self.embedding_part = tf.multiply(self.embedding_index, tf.reshape(self.feat_value, [-1, self.field_size, 1]))
        
        # FM部分一阶特征
        # None * F*1
        self.embedding_first = tf.nn.embedding_lookup(self.weight['first_order'],
                                                      self.feat_index)
        #[None*F*1].*[None*F*1] = None*F*1
        self.embedding_first = tf.multiply(self.embedding_first, tf.reshape(self.feat_value, [-1, self.field_size, 1]))
        # None*F
        self.first_order = tf.reduce_sum(self.embedding_first, 2) 

        # 二阶特征 None*K
        self.sum_second_order = tf.reduce_sum(self.embedding_part, 1)
        self.sum_second_order_square = tf.square(self.sum_second_order)
        self.square_second_order = tf.square(self.embedding_part)
        self.square_second_order_sum = tf.reduce_sum(self.square_second_order, 1) 
        # 1/2*((a+b)^2 - a^2 - b^2)=ab
        # None*K
        self.second_order = 0.5 * tf.subtract(self.sum_second_order_square, self.square_second_order_sum)

        # FM部分的输出 None*(F+K)
        self.fm_part = tf.concat([self.first_order, self.second_order], axis=1)

        # DNN部分
        # None*(F*K)
        self.deep_embedding = tf.reshape(self.embedding_part, [-1, self.field_size * self.embedding_size])

        # 全连接部分
        for i in range(0, len(self.deep_layers)):
            self.deep_embedding = tf.add(tf.matmul(self.deep_embedding, self.weight["layer_%d" % i]),
                                         self.weight["bias_%d" % i])
            # self.deep_embedding =tf.matmul(self.deep_embedding, self.weight["layer_%d" % i])
            self.bn_out = tf.layers.batch_normalization(self.deep_embedding, training=self.is_training)
            # self.bn_out = tf.layers.dropout(self.deep_embedding, rate=self.keep_prob,training=self.is_training)
            self.deep_embedding = self.activate(self.bn_out)
            self.deep_embedding = tf.layers.dropout(self.deep_embedding, rate =1.0-self.keep_prob, training= self.is_training)

        # FM输出与DNN输出拼接 None*(F+K+layer[-1]])
        din_all = tf.concat([self.fm_part, self.deep_embedding], axis=1)
        #None*1
        self.out = tf.add(tf.matmul(din_all, self.weight['last_layer']), self.weight['last_bias'])
        
        #3. ------------------确定损失---------------------------------------
        # loss部分 None*1
        self.prob = tf.nn.sigmoid(self.out)
        # self.entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= self.label, logits= self.out))
        # self.entropy_loss = -tf.reduce_mean(
        #     self.label * tf.log(tf.clip_by_value(self.prob, 1e-10, 1.0))+ (1 - self.label)* tf.log(tf.clip_by_value(1-self.prob,1e-10,1.0)))
        self.entropy_loss = focal_loss(self.prob, self.label, alpha=0.5, gamma=2)
        # self.entropy_loss = weighted_binary_crossentropy(self.prob, self.label, pos_ratio=self.pos_ratio)

        # 正则：sum(w^2)/2*l2_reg_coef
        
        self.reg_loss = tf.contrib.layers.l2_regularizer(self.l2_reg_coef)(self.weight["last_layer"])
        for i in range(len(self.deep_layers)):
            self.reg_loss += tf.contrib.layers.l2_regularizer(self.l2_reg_coef)(self.weight["layer_%d" % i])
            # tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(self.l2_reg_coef)(self.weight['layer_1']))
        # print(self.entropy_loss.shape.as_list(), self.reg_loss.shape.as_list())
        self.loss = self.entropy_loss + self.reg_loss
        
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,3000, 0.99,staircase=False)
        opt = tf.train.AdamOptimizer(self.learning_rate)
        # opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        with tf.control_dependencies(update_ops):
            # self.train_op = opt.minimize(self.loss, global_step = self.global_step)
            self.train_op = opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)
        self.saver = tf.train.Saver(max_to_keep=3)

    def train(self, sess, feat_index, feat_value, label):
        _, step = sess.run([self.train_op, self.global_step], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value,
            self.label: label,
            self.keep_prob: self.keep_prob_v,
            self.is_training:True})
        return  step

    def predict(self, sess, feat_index, feat_value, batch_size=None):
        if batch_size is None:
            prob = sess.run([self.prob], feed_dict={
                self.feat_index: feat_index,
                self.feat_value: feat_value,
                self.keep_prob: 1,
                self.is_training:False})[0]
        else:
            data =Dataset(feat_value, feat_index, [None]*len(feat_index), batch_size, shuffle=False)
            probs =[]
            for feat_index, feat_value, _ in data:
                prob = sess.run([self.prob], feed_dict={
                self.feat_index: feat_index,
                self.feat_value: feat_value,
                self.keep_prob: 1,
                self.is_training:False})[0]
                probs.append(prob.ravel())

            prob = np.concatenate(probs)

        return prob.ravel()
    
    def evaluate(self, sess, feat_index, feat_value, label, batch_size=None):
        tloss, entloss,regloss = 0,0,0
        if batch_size is None:
            tloss, entloss,regloss = sess.run([self.loss, self.entropy_loss, self.reg_loss],feed_dict={
                                            self.feat_index: feat_index,
                                            self.feat_value: feat_value,
                                            self.label: label,
                                            self.keep_prob: 1,
                                            self.is_training:False})
        else:
            data = Dataset(feat_value,feat_index,label, batch_size, shuffle=False)
            for i, (feat_index, feat_value, label) in enumerate(data,1):
                _tloss, _entloss, _regloss = sess.run([self.loss, self.entropy_loss, self.reg_loss],feed_dict={
                                                self.feat_index: feat_index,
                                                self.feat_value: feat_value,
                                                self.label: label,
                                                self.keep_prob: 1,
                                                self.is_training:False})
                tloss  = tloss+ (_tloss-tloss)/i 
                entloss = entloss + (_entloss-entloss)/i
                regloss = regloss + (_regloss-regloss)/i

        return tloss, entloss, regloss

    def save(self, sess, path, global_step):
        if self.saver is not None:
            self.saver.save(sess, save_path=path, global_step= global_step)

    def restore(self, sess, path):
        model_file = tf.train.latest_checkpoint(path)
        if model_file is not None:
            print('restore model:', model_file)
            self.saver.restore(sess, save_path=model_file)



if __name__ == '__main__':

    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    params ={'feature_size':None,
         'field_size':None,
         'embedding_size':4,
         'deep_layers':[32,32,32],
         'epoch':200,
         'batch_size':128,
         'learning_rate':0.001,
         'l2_reg': 0.001,
         'keep_prob':0.7,
         'checkpoint_dir':os.path.join(BASE_PATH,'data/deepfm'),
         'training_model':True}

    with tf.Session() as sess:
        model = DeepFM(params)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer()) # global_step counter etc.
        sys.stdout.flush()
        
        if params['training_model']:
            #---------------training---------------------------------
            for i in range(params['epoch']):
                print('epoch ={}'.format(i).center(50,'-'))
                for j, (xi, xv, y) in enumerate(train_data):
                    loss,_,  step = model.train(sess, xi, xv, y)
                    if j %1000 ==0:
                      
                        train_loss,train_entropy,train_reg = model.evaluate(sess, Xi,Xv, Y)
                        val_loss,val_entropy, val_reg = model.evaluate(sess, val_Xi, val_Xv, val_y)                    
                        print('---batch= %d--- \n train_loss=%f,\t train_entropy=%f,\t train_reg=%f \n  val_loss=%f,\t val_entropy=%f,\t val_reg=%f' % (
                               j,train_loss,train_entropy,train_reg, val_loss,val_entropy,val_reg))
                if i%10 ==0 or i == params['epoch']-1:
                    model.save(sess, model.checkpoint_dir, i)
                    prob = model.predict(sess, Xi, Xv)
                    hit_rate, top_k = top_ratio_hit_rate(np.array(Y).ravel(), np.array(prob[0]).ravel(), top_ratio=0.001) # ravel return view, flatten return copy
                    print('top-k={}, train-hit-rate={}'.format(top_k ,hit_rate))
                    
                    #-----------------test-----------------------------------
                    probs =[]
                    test_y=[]
                    for xi, xv, y in test_data:
                        prob = model.predict(sess, xi, xv)  # list of np.ndarry
                        probs.extend(prob[0].ravel().tolist())
                        test_y.extend(y.tolist())
                    hit_rate, top_k = top_ratio_hit_rate(np.array(test_y).ravel(), np.array(probs).ravel(), top_ratio=0.001)
                    print('top-k={}, test-hit-rate={}'.format(top_k ,hit_rate))
                    calc_threshold_vs_depth(np.asarray(test_y).ravel(), np.asarray(probs).ravel())
            
        else:
            model.restore(sess, os.path.split(model.checkpoint_dir)[0])
            
            probs=[]
            Y =[]
            for xi, xv, y in train_data:
                prob = model.predict(sess, xi, xv)  # np.ndarry
                probs.extend(prob[0].ravel().tolist())
                Y.extend(y.tolist())
            hit_rate, top_k = top_ratio_hit_rate(np.array(Y).ravel(), np.array(probs).ravel(), top_ratio=0.001)
            print('train-top-k={}, train-hit-rate={}'.format(top_k ,hit_rate))
            probs=[]
            test_y=[]
            for xi, xv, y in test_data:
                prob = model.predict(sess, xi, xv)  # np.ndarry
                probs.extend(prob[0].ravel().tolist())
                test_y.extend(y.tolist())
            hit_rate, top_k = top_ratio_hit_rate(np.array(test_y).ravel(), np.array(probs).ravel(), top_ratio=0.001)
            print('test-top-k={}, test-hit-rate={}'.format(top_k ,hit_rate))
