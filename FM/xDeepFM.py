import os
import numpy as np
import tensorflow as tf 

class xDeepFM(object):
    def __init__(self, config, seed=2019):
        tf.set_random_seed(seed)
        self._feature_size = config.pop('feature_size')
        self._field_size = config.pop('field_size')
        self.embedding_size = config.pop('embedding_size')
        self.keep_prob = config.pop('keep_prob')
        self.cross_layer_sizes = config.pop('cross_layer_sizes')
        self.dnn_layer_sizes = config.pop('dnn_layer_sizes')
        self._check_config(config)
        self.config = config
        self.logits = self.build()
        
    def _check_config(self,config):
        if isinstance(config, dict):
            keys = set(config.keys())
            allow_keys=['cin_filter_latent_dim','cin_activate','cin_resblock_hidden_size','cin_resblock_hidden_activate']
            diffs = keys.difference(allow_keys)
            if diffs:
                raise ValueError('unknown configuratios:{}'.format(diffs))              
        else:
            raise TypeError('config should be dict')

    def _init_placeholders(self):
        self.feat_value = tf.placeholder(shape=[None,None], dtype= tf.float32)
        self.feat_index = tf.placeholder(shape=[None,None], dtype=tf.int32)
        self.label = tf.placeholder(shpae=[None,1],dtype=tf.float32)
        self.keep_prob = tf.placeholder(shape=[], dtype=tf.float32,name='keep_prob')
        self.is_training = tf.placeholder(tf.bool, shape=[],name='is_training')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def build(self):
        use_resblock= self.config.get('use_resblock',False)
        split_connect self.config.get('split_connect',True)
        reduce_filter_complexity = self.config.get('reduce_filter_complexity',True)
        add_bias = self.config.get('add_bias',False)

        self._init_placeholders()
        emb_out, emb_merge_size = self._build_embedding()
        linear_logits = self._build_linear() 
        cin_logits = self._build_cin(emb_out, use_resblock=use_resblock, split_connect=split_connect, 
                                    reduce_filter_complexity=reduce_filter_complexity,add_bias=add_bias)
        dnn_logits = self._build_dnn(emb_out)
        logits = tf.add_n([linear_logits,cin_logits,dnn_logits])
        self.prob = tf.nn.sigmoid(logits)

        self.loss, self.entropy_loss, self.reg_loss = self.compute_loss()
        self.train_op = self.backward()
        self.saver = tf.train.Saver(max_to_keep=3)
        return logits

    def _build_embedding(self):
        #None: batch_sze
        #F: field_size
        #k: embedding_size
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

    def _build_cin(self, emb_out, use_resblock=False, split_connect=True, reduce_d=False, add_bias=False):
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

                if reduce_d:
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
                activate = tf.config.get('cin_resblock_hidden_activate',tf.nn.relue)
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

    def compute_loss(self):
        labels = tf.reshape(self.label,[-1])
        logits = tf.reshape(self.logits, [-1])
        cel = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        #reg loss
        params = tf.get_collections(tf.GraphKeys.TRAINABLE_VARIABLES)
        params = [p for p in params if 'bias' not in p.name]
        l2_reg = self.config.get('l2_reg',0.01)
        regloss = tf.add_n([tf.contrib.layers.l2_regularizer(l2_reg)(p) for p in params])
        loss = cel + regloss
        return loss, cel, regloss
    
    def backward(self):
        lr = tf.train.exponential_decay(self.learning_rate, self.global_step, 3000, 0.99,staircase=False)
        opt = tf.train.AdamOptimizer(lr)
        train_vars = tf.trainable_variables()
        update_ops = tf.collections(tf.GraphKeys.UPDATE_OPS)
        gradients = tf.gradient(self.loss, train_vars) # sum for params but not for input 
        clip_grads,_ = tf.clip_by_global_norm(gradients,5)
        with tf.control_dependencies(update_ops):
            train_op = opt.minimize(self.loss, global_step = self.global_step)
            #train_op = opt.apply_gradients(zip(clip_gradients, train_vars), global_step = self.global_step)
        return train_op









            


