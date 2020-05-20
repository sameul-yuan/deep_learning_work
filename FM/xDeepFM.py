import os
import numpy as np
import tensorflow as tf 
from losses import focal_loss,weighted_binary_crossentropy
from utils import Dataset

class xDeepFM(object):
    def __init__(self, config, seed=2019):
        tf.set_random_seed(seed)
        self._feature_size = config.pop('feature_size')
        self._field_size = config.pop('field_size')
        self.embedding_size = config.pop('embedding_size')
        self.feed_keep_prob = config.pop('keep_prob')
        self.cross_layer_sizes = config.pop('cross_layer_sizes')
        self.dnn_layer_sizes = config.pop('dnn_layer_sizes')
        self.learning_rate = config.pop('learning_rate')
        self.checkpoint_dir = config.pop('checkpoint_dir')
        # self._check_config(config)
        self.config = config
        self.build()
        
    def _check_config(self,config):
        if isinstance(config, dict):
            keys = set(config.keys())
            allow_keys=['cin_filter_latent_dim','cin_activate','cin_resblock_hidden_size','cin_resblock_hidden_activate','l2_reg',
                        'use_resblock','split_connect','reduce_filter_complexity','add_bias']
            diffs = keys.difference(allow_keys)
            if diffs:
                raise ValueError('unknown configuratios:{}'.format(diffs))              
        else:
            raise TypeError('config should be dict')

    def _init_placeholders(self):
        self.feat_value = tf.placeholder(shape=[None,None], dtype= tf.float32)
        self.feat_index = tf.placeholder(shape=[None,None], dtype=tf.int32)
        self.label = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.keep_prob = tf.placeholder(shape=[], dtype=tf.bool,name='keep_prob')
        self.is_training = tf.placeholder(tf.bool, shape=[],name='is_training')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def _build_embedding(self):
        #None: batch_sze
        #F: field_size
        #k: embbeding_size
        with tf.variable_scope('embedding') as scope:
            self.fm_embedding = tf.get_variable('embedding', shape=[self._feature_size, self.embedding_size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1),dtype=tf.float32)
            emb = tf.nn.embedding_lookup(self.fm_embedding, self.feat_index) # None*F*K
            feat_value = tf.reshape(self.feat_value, shape=[-1,self._field_size, 1]) # None*F*1
            emb_out = tf.multiply(emb, feat_value) # None*F*K
            emb_merge_size = self._field_size * self.embedding_size
            emb_out = tf.reshape(emb_out, shape=[-1, emb_merge_size]) # None* (F*K)
        return emb_out, emb_merge_size

    def _build_linear(self):
        with tf.variable_scope('linear',initializer=tf.truncated_normal_initializer(stddev=0.1)) as scope:
            self.linear_embedding = tf.get_variable('embedding',shape=[self._feature_size,1],dtype=tf.float32)
            self.linear_bias = tf.get_variable('bias', shape=[1],initializer=tf.zeros_initializer(),dtype=tf.float32)

            w = tf.nn.embedding_lookup(self.linear_embedding, self.feat_index) # None *F*1
            feat_value = tf.reshape(self.feat_value,shape=[-1,self._field_size,1])
            xw = tf.multiply(w, feat_value) # None*F*1
            xw = tf.reduce_sum(xw, axis=[1]) #None *1
            linear_out = tf.add(xw,self.linear_bias)
        return linear_out # None * 1

    def _build_cin(self, emb_out, use_resblock=False, split_connect=True, reduce_filter_complexity=False, add_bias=False):
        # hk: field_size of next layer
        h0 = self._field_size
        hk = h0
        cin_x0 = tf.reshape(emb_out, shape=[-1,self._field_size, self.embedding_size])
        cin_xk = cin_x0 #cin input of next layer
        cin_outs = []
        cin_out_size = 0
        cin_x0_split = tf.split(cin_x0,self.embedding_size*[1],2) # k# None*F*1
        with tf.variable_scope('cin',initializer=tf.truncated_normal_initializer(stddev=0.1)) as scope:
            for index, next_hk in enumerate(self.cross_layer_sizes):
                cin_xk_split = tf.split(cin_xk, self.embedding_size,2) # k# None*hk*1
                dots = tf.matmul(cin_x0_split, cin_xk_split, transpose_b=True) # k*None*F*hk
                dots = tf.reshape(dots, shape=[self.embedding_size,-1, h0*hk]) # k*None* (F*hk)
                dots = tf.transpose(dots, perm=[1,0,2]) # None * k *(F*hk)

                if reduce_filter_complexity:
                    latent_dim = self.config.get('cin_filter_latent_dim',2)
                    filter0 = tf.get_variable('filter0_'+str(index),
                                shape=[1,next_hk, h0,latent_dim],dtype=tf.float32)
                    filter1 = tf.get_variable('filter1_'+str(index),
                                shape=[1, next_hk,latent_dim, hk],dtype=tf.float32)
                    filters = tf.matmul(filter0, filter1) # 1*next_hk*F*hk
                    filters = tf.reshape(filters,shape=[1,next_hk, h0*hk])
                    filters = tf.transpose(filters, perm=[0,2,1]) # 1*(h0*hk)*next_hk
                else:
                    filters = tf.get_variable('filter_'+str(index),
                                shape=[1,h0*hk,next_hk],dtype=tf.float32)
                # None * k *(F*hk) conv 1*(h0*hk)*next_hk = None*k*next_hk
                layer_out = tf.nn.conv1d(dots,filters=filters,stride=1,padding='VALID') # None*k*next_hk

                if add_bias:
                    bias = tf.get_variable('filter_bias'+str(index),
                                    shape=[next_hk],dtype= tf.float32, initializer=tf.zeros_initializer())
                    layer_out = tf.nn.bias_add(layer_out, bias)

                activate = self.config.get('cin_activate',tf.nn.relu)
                layer_out = activate(layer_out)
                layer_out = tf.transpose(layer_out, perm=[0,2,1]) #None*next_hk*k

                if split_connect:
                    if index != len(self.cross_layer_sizes)-1:
                        cin_xk, cin_out = tf.split(layer_out, 2*[int(next_hk/2)],1)
                        cin_out_size += int(next_hk/2)
                    else:
                        cin_xk = 0
                        cin_out = layer_out
                        cin_out_size += next_hk
                    hk = int(next_hk/2)
                else:
                    cin_xk = layer_out
                    cin_out = layer_out
                    cin_out_size += next_hk
                cin_outs.append(cin_out)
            result = tf.concat(cin_outs,axis=1) # None*cin_out_size * k
            result = tf.reduce_sum(result, axis=-1) # None * cin_out_size

            if use_resblock:  # concat instead of add
                hidden_size = self.config.get('cin_resblock_hidden_size',32)
                block_w = tf.get_variable('resblock_hidden_w',shape=[cin_out_size, hidden_size],dtype=tf.float32)
                block_b = tf.get_variable('resblock_hidden_b',shape=[hidden_size],dtyep=tf.float32,initializer=tf.zeros_initializer())
                hidden_input = tf.nn.xw_plus_b(result, block_w, block_b)
                activate = tf.config.get('cin_resblock_hidden_activate',tf.nn.relu)
                hidden_out = activate(hidden_input) #None* hidden_size

                merge_w = tf.get_variable('resblock_merge_w', shape=[hidden_size+cin_out_size,1],dtype=tf.float32)
                merge_b = tf.get_varaible('resblock_merge_b',shape=[1],dtype=tf.float32,initializer=tf.zeros_initializer())
                merge_input= tf.concat([hidden_out, result], axis=1) # None*(hidden_size+cin_out_size)
                xdeep_out = tf.nn.xw_plus_b(merge_input,merge_w, merge_b)
            else:
                w = tf.get_variable('cin_w',shape=[cin_out_size,1],dtype=tf.float32)
                b = tf.get_variable('cin_b',shape=[1],dtype=tf.float32, initializer=tf.zeros_initializer())
                xdeep_out = tf.nn.xw_plus_b(result, w,b)
            
            return xdeep_out # None * 1
    
    def _build_dnn(self, emb_out):
        nn_input = emb_out  # None * emb_merge_size
        with tf.variable_scope("dnn") as scope:
            for index, layer_size in enumerate(self.dnn_layer_sizes):
                nn_input = tf.layers.dense(nn_input, layer_size, activation=None,
                                kernel_initializer=tf.variance_scaling_initializer(scale=2.0,mode='fan_in'))
                nn_input = tf.layers.batch_normalization(nn_input)
                nn_input = tf.nn.relu(nn_input)
            
            #output layer
            output = tf.layers.dense(nn_input,1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        return output #None*1

    def build(self):
        use_resblock = self.config.get('use_resblock',False)
        split_connect = self.config.get('split_connect',True)
        reduce_filter_complexity = self.config.get('reduce_filter_complexity',False)
        add_bias = self.config.get('add_bias',False)

        self._init_placeholders()
        emb_out, emb_merge_size = self._build_embedding()
        linear_logits = self._build_linear() 
        cin_logits = self._build_cin(emb_out, use_resblock=use_resblock, split_connect=split_connect, 
                                    reduce_filter_complexity=reduce_filter_complexity, add_bias=add_bias)
        dnn_logits = self._build_dnn(emb_out)
        self.logits = tf.add_n([linear_logits,cin_logits,dnn_logits])
        self.prob = tf.nn.sigmoid(self.logits)

        self.loss, self.entropy_loss,self.reg_loss = self.compute_loss()

        self.train_op = self.backward()
        self.saver = tf.train.Saver(max_to_keep=3)
        

    def compute_loss(self):
        labels = tf.reshape(self.label, [-1])
        logits = tf.reshape(self.logits, [-1])
        ce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

        #regularization loss
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        params = [param for param in params if 'bias' not in param.name]
        l2_reg = self.config.get('l2_reg', 0.001)
        reg_loss =tf.add_n([tf.contrib.layers.l2_regularizer(l2_reg)(param) for param in params])
        
        loss = ce_loss + reg_loss
        return loss, ce_loss, reg_loss

    def backward(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,3000, 0.99, staircase=False)
        opt = tf.train.AdamOptimizer(learning_rate)
        # opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        trainable_params = tf.trainable_variables()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        gradients = tf.gradients(self.loss, trainable_params) # auto divide batch_size for param, but not for input
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(self.loss, global_step = self.global_step)
            # train_op = opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)    
        return train_op

    def train(self, sess, feat_index, feat_value, label):
        _, step = sess.run([self.train_op, self.global_step], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value,
            self.label: label,
            self.keep_prob: self.feed_keep_prob,
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


            


