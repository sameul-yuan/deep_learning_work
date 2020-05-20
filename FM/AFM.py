import os
import itertools 
import numpy as np
import tensorflow as tf
from utils import Dataset

class AFM(object):
    """attention FM"""
    def __init__(self,config,seed=42):
        tf.set_random_seed(seed)
        self._feature_size = config.pop('feature_size')
        self._field_size = config.pop('field_size')
        self.embedding_size = config.pop('embedding_size')
        self.attention_factor=config.pop('attention_factor',4)
        self.feed_keep_prob = config.pop('keep_prob')
        self.dnn_layer_sizes = config.pop('dnn_layer_sizes')
        self.learning_rate = config.pop('learning_rate')
        self.l2_reg = config.pop('l2_reg',0.01)
        self.checkpoint_dir = config.pop('checkpoint_dir')
        self._validate_config(config)
        self.config = config
        self.logits = self.build()

    def _validate_config(self,config):

        #TODO(yyq)
        allow_kargs={}

    def _init_placeholders(self):
        self.feat_value = tf.placeholder(shape=[None,None], dtype= tf.float32)
        self.feat_index = tf.placeholder(shape=[None,None], dtype=tf.int32)
        self.label = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.keep_prob = tf.placeholder(shape=[], dtype=tf.float32,name='keep_prob')
        self.is_training = tf.placeholder(tf.bool, shape=[],name='is_training')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        
    def _build_embedding(self):
        #None: batch_sze
        #F: field_size
        #k: embedding_size
        with tf.variable_scope('embedding') as scope:
            self.fm_embedding = tf.get_variable('embedding', shape=[self._feature_size, self.embedding_size],
                                    initializer=tf.truncated_normal_initializer(stddev=0.05),dtype=tf.float32)
            emb = tf.nn.embedding_lookup(self.fm_embedding, self.feat_index) # None*F*K
            feat_value = tf.reshape(self.feat_value, shape=[-1,self._field_size, 1]) # None*F*1
            emb_out = tf.multiply(emb, feat_value) # None*F*K
            # emb_merge_size = self._field_size * self.embedding_size
            # emb_out = tf.reshape(emb_out, shape=[-1, emb_merge_size]) # None* (F*K)
        return emb_out
    
    def _build_linear(self):
        with tf.variable_scope('linear',initializer=tf.truncated_normal_initializer(stddev=0.05)) as scope:
            self.linear_embedding = tf.get_variable('embedding',shape=[self._feature_size,1],dtype=tf.float32,)
            self.linear_bias = tf.get_variable('bias', shape=[1],initializer=tf.zeros_initializer(),regularizer=None,dtype=tf.float32)

            w = tf.nn.embedding_lookup(self.linear_embedding, self.feat_index) # None *F*1
            feat_value = tf.reshape(self.feat_value,shape=[-1,self._field_size,1])
            xw = tf.multiply(w, feat_value) # None*F*1
            xw = tf.reduce_sum(xw, axis=[1]) #None *1
            linear_out = tf.add(xw,self.linear_bias)
        return linear_out # None * 1

    def _build_attention(self, use_dnn=False):
         # glorot_normal = np.sqrt(2.0/(self.embedding_size+self.attention_factor))
        with tf.variable_scope('attention',initializer=tf.truncated_normal_initializer(stddev=0.05)) as scope:
            self.att_w = tf.get_variable('att_w',shape=[self.attention_factor,self.embedding_size],dtype=tf.float32,
                                            initializer= tf.glorot_normal_initializer(),
                                            regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg))
            self.att_b = tf.get_variable('att_b',shape=[self.attention_factor],dtype=tf.float32,initializer=tf.zeros_initializer())
            self.project_h  = tf.get_variable('proj_h',shape=[self.attention_factor,1],dtype=tf.float32)
            self.project_p = tf.get_variable('proj_p',shape=[self.embedding_size,1],dtype=tf.float32)
        
        vecs = [tf.gather(self.emb_out,indices=k,axis=1) for k in range(self._field_size)] #indices=[k] None*1*emb; indices=k,None*emb
        rows=[]
        cols=[]
        for r,c in itertools.combinations(vecs,2):
            rows.append(r)   #None*emb
            cols.append(c)
        p = tf.stack(rows,axis=1) #None*k(k-1)/2*emb
        q = tf.stack(cols,axis=1) #None*k(k-1)/2*emb
        inp = tf.multiply(p,q) #None*k(k-1)/2*emb
        print(inp.shape)

        # element_wise_product =[]
        # for i in range(self._field_size):
        #     for j in range(i+1, self._field_size):
        #         element_wise_product.append(tf.multiply(self.emb_out[:,i,:],self.emb_out[:,j,:])) #None*emb
        # inp = tf.stack(element_wise_product,axis=1) #None*k(k-1)/2 *emb
     
        # (wx+b) ->relu(wx+b) ->h*relu(wx+b)
        wx = tf.tensordot(inp, self.att_w,axes=(-1,-1)) #None * k(k-1)/2 * factor
        wx_plus_b  = tf.nn.relu(tf.nn.bias_add(wx,self.att_b)) #None*k(k-1)/2*factor
        attenton_score = tf.tensordot(wx_plus_b, self.project_h,axes=(-1,0)) # None* k(k-1)/2* 1
        attention_coef = tf.nn.softmax(attenton_score,axis=1) #None* k(k-1)/2* 1
        # att_out = tf.reduce_sum(attention_coef*tf.stop_gradient(inp),axis=1) #None*emb
        att_out = tf.reduce_sum(attention_coef*inp,axis=1) #None*emb
        att_out = tf.layers.dropout(att_out, rate=1-self.keep_prob, training=self.is_training)
        # original
        att_fm = tf.tensordot(att_out, self.project_p,  axes=1) #None*1
        # feed to dnn
    
        nn_input = att_out
        for index, layer_size in enumerate(self.dnn_layer_sizes):
            nn_input = tf.layers.dense(nn_input, layer_size, activation=None,
                            kernel_initializer=tf.variance_scaling_initializer(scale=2.0,mode='fan_in'),   
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
            # nn_input = tf.layers.batch_normalization(nn_input)
            nn_input = tf.nn.relu(nn_input)
            nn_input = tf.layers.dropout(nn_input, rate=1-self.keep_prob,training=self.is_training)
            #output layer
        dnn_out = tf.layers.dense(nn_input,1, activation=None,kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)) #None*1


        return  att_fm, dnn_out

    def build(self):
        use_dnn = self.config.get('use_dnn',False)
        combiner = self.config.get('combiner','ANFM')
        self._init_placeholders()
        self.emb_out = self._build_embedding()
        linear_logits = self._build_linear()
        att_logits, dnn_logits = self._build_attention(use_dnn=use_dnn)
        if combiner=='AFM':
            self.logits = tf.add_n([linear_logits, att_logits])
        elif combiner=='ANFM':
            self.logits = tf.add_n([linear_logits, dnn_logits])
        elif combiner=='ADeepFM':
            self.logits = tf.add_n([linear_logits, att_logits, dnn_logits])

        self.prob = tf.nn.sigmoid(self.logits)

        self.loss, self.entropy_loss,self.reg_loss = self.compute_loss(self.label,self.logits)

        self.train_op = self.backward()
        self.saver = tf.train.Saver(max_to_keep=3)

    def compute_loss(self,labels, logits):
        labels = tf.reshape(labels, [-1])
        logits = tf.reshape(logits, [-1])
        ce_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

        # regularization loss
        # reg_tensors= tf.get_collection( tf.GraphKeys.REGULARIZATION_LOSSES)
        # print(reg_tensors)
        # reg_loss = tf.add_n([tensor for tensor in reg_tensors])
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # params = [param for param in params if 'bias' not in param.name]
        # l2_reg = self.config.get('l2_reg', 0.001)
        # reg_loss = tf.add_n([tf.contrib.layers.l2_regularizer(l2_reg)(param) for param in params])
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
        _, step,prob = sess.run([self.train_op, self.global_step,self.prob], feed_dict={
            self.feat_index: feat_index,
            self.feat_value: feat_value,
            self.label: label,
            self.keep_prob: self.feed_keep_prob,
            self.is_training: True})
        return step,prob

    def predict(self, sess, feat_index, feat_value, batch_size=None):
        if batch_size is None:
            prob = sess.run([self.prob], feed_dict={
                self.feat_index: feat_index,
                self.feat_value: feat_value,
                self.keep_prob: 1,
                self.is_training: False})[0]
        else:
            data = Dataset(feat_value, feat_index, [None] * len(feat_index), batch_size, shuffle=False)
            probs = []
            for feat_index, feat_value, _ in data:
                prob = sess.run([self.prob], feed_dict={
                    self.feat_index: feat_index,
                    self.feat_value: feat_value,
                    self.keep_prob: 1,
                    self.is_training: False})[0]
                probs.append(prob.ravel())

            prob = np.concatenate(probs)

        return prob.ravel()

    def evaluate(self, sess, feat_index, feat_value, label, batch_size=None):
        tloss, entloss, regloss = 0, 0, 0
        if batch_size is None:
            tloss, entloss, regloss = sess.run([self.loss, self.entropy_loss, self.reg_loss], feed_dict={
                self.feat_index: feat_index,
                self.feat_value: feat_value,
                self.label: label,
                self.keep_prob: 1,
                self.is_training: False})
        else:
            data = Dataset(feat_value, feat_index, label, batch_size, shuffle=False)
            for i, (feat_index, feat_value, label) in enumerate(data, 1):
                _tloss, _entloss, _regloss = sess.run([self.loss, self.entropy_loss, self.reg_loss], feed_dict={
                    self.feat_index: feat_index,
                    self.feat_value: feat_value,
                    self.label: label,
                    self.keep_prob: 1,
                    self.is_training: False})
                tloss = tloss + (_tloss - tloss) / i
                entloss = entloss + (_entloss - entloss) / i
                regloss = regloss + (_regloss - regloss) / i

        return tloss, entloss, regloss

    def save(self, sess, path, global_step):
        if self.saver is not None:
            self.saver.save(sess, save_path=path, global_step=global_step)

    def restore(self, sess, path):
        model_file = tf.train.latest_checkpoint(path)
        if model_file is not None:
            print('restore model:', model_file)
            self.saver.restore(sess, save_path=model_file)
        
        