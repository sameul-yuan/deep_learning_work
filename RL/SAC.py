#! -*- coding: utf-8 -*- 

import os 
import sys
import warnings
import atexit
import argparse

import pandas as pd 
import numpy as np 
import tensorflow as tf 
from tqdm import tqdm 
from tensorflow import keras 

from sac_core import GaussianPolicy 
from min_heap import MinHeap
from nn_modules import get_session_config
from logger import Logger 
from dataset import Dataset,CsvBuffer
from metrics import AucROC, WeightedBinaryCrossEntropy, focal_loss
from sklearn.metrics import roc_auc_score 

warngings.filterwarnings(aciton='ignore',category=FutureWarning)

class SAC(object):
    def __init__(self, n_state,n_action, a_scale=5, log_std_max=1, log_std_min=-10, alpha=0.2, 
                    actor_lr=0.001, critic_lr=0.001, vf_lr=0.001, polyavk=0.995, discount=0.99, 
                    assess_interval=10, save_interval=5000, buffer_size=20000,
                    logger=None, checkpoint_queen=None,**kwargs):
        self.logger = logger 
        self.logger.save_config(locals())
        self.n_state = n_state
        self.n_action = n_state
        self.a_scale= a_scale
        self.log_std_max = log_std_max 
        self.log_std_min = log_std_min
        self.alpha = alpha  #weight of entropy 
        self.actor_lr = actor_lr 
        self.critic_lr = critic_lr
        self.vf_lr = vf_lr 
        self.polyavk = polyavk #smooth factor for updating target value 
        self.discount = discount
        self.kwargs = kwargs
        self.assess_interval = assess_interval
        self.save_interval = save_interval
        self.buffer = MemoryBuffer(buffer_size, with_per=False)
        self.ckpt_queen = checkpoint_queen
        self.prefix = self.__class__.__name__.lower()
        self._init_model()

    def __init_model(self):
        policy = GaussianPolicy(self.n_state, self.n_action, a_scale=self.a_scale, log_std_max = self.log_std_max, 
                                log_std_min =self.log_std_min, **self.kwargs):
        self.actor = policy.build_actor(name_scope='actor')
        self.critic1 = policy.build_critic(name_scope='critic1')
        self.critic2 = policy.build_critic(name_scope='critic2')
        self.vf= policy.build_vf(name_scope='vf')
        self.target_vf = policy.build_vf(name_scope='target_vf')
        self.copy_weghts()
        self._init_optimizer()

    def _init_optimizer(self):
        #maximize Q1(st,a)-alpha*log(pa), a=current action
        grad_ys = K.placeholder(shape=(None,self.n_action)) # dQ/da
        states = self.actor.inputs[0]
        mu,pi, logp_pi = self.actor.outputs #self.actor(states)
        #p_loss = K.mean(self.alpha*logp_pi -self.critic1([states,pi])) # critic is relevant to actor input 
        p_loss = K.mean(self.alpha*logp_pi - grad_ys*pi)
        p_optimizer = keras.optimizers.Adam(self.actor_lr)
        p_updates = p_optimizer.get_updates(params= self.actor.trainable_weights, loss=p_loss)
        self.p_train = K.function(inputs=[states,grad_ys],outputs=[mu,pi,p_loss],updates=p_updates)
        self.action_grads=K.function(inputs=[self.critic1.input[0],self.critic1.input[1]], 
                                    outputs=K.gradients(self.cirtic1.output, self.critic1.input[1])) #(batch, len(xs))
        
        # minimize (Q(st,at)-(r(st,at)+gamma*V'(st+1)))**2
        target_q = K.placeholder(shape=(None,1))
        q1_loss = K.mean(keras.losses.mse(target_q, self.critic1.outputs[0]))
        q2_loss = K.mean(keras.losses.mse(target_q, self.critic2.outputs[0]))
        q1_optimizer = keras.optimizers.Adam(self.critic_lr)
        q2_optimizer = keras.optimizers.Adam(self.critic_lr)
        q1_updates = q1_optimizer.get_updates(params=self.critic1.trainable_weights,loss=q1_loss)
        q2_updates = q2_optimizer.get_updates(params=self.critic2.trainable_weights,loss=q2_loss)
        q1_updates += self.critic1.updates
        q2_updates += self.critic2.updates
        print(self.critic1.input+[target_q])
        self.q1_train = K.function(inputs=self.critic1.input+[target_q], outputs=[q1_loss],updates= q1_updates)
        self.q2_train = K.function(inputs=self.critic2.input+[target_q],outputs=[q2_loss],updates=q2_updates)

        # minimize (V(st) -(Q1(st,a)-alpha*log(pa)))**2 
        target_v = K.placeholder(shape=(None,1))
        v_loss = K.mean(keras.losses.mse(target_v, self.vf.outputs[0]))
        v_optimizer = keras.optimizers.Adam(self.vf_lr)
        v_updates= v_optimizer.get_updates(params=self.vf.trainable_weights, loss=v_loss)
        v_updates += self.vf.updates

        self.v_train = K.function(inputs=[self.vf.inputs[0],target_v],outputs=[v_loss],updates=v_updates) #output before update

    def _copy_weights(self):
        w, target_w = self.vf.get_weights(), self.target_vf.get_weights()
        for i in range(len(w)):
            target_w[i] = self.polyavk*target_w + (1.- self.polyavk)*w[i]
        self.target_vf.set_weights(target_w)

    def update_model(self,states, actions, target_q):
        mu,pi,logp_pi = self.actor.predict(states)
        grad_ys = self.action_grads([states,pi])[0]
        _,_,p_loss = self.p_train([states, grad_ys]) #depend on Q1

        q1_loss = self.q1_train([states, actions, target_q])
        q2_loss = self.q2_train([states,actions,target_q])
        
        q1 = self.critic1.predict([states,pi]) #current pi 
        q1 = self.critic1.predict(pstates,pi)
        target_v = np.minimum(q1,q2) - self.alpha*logp_pi
        v_loss = self.v_train([states,target_v]) #depend on q1,q2 and actor 

        self._copy_weights()
        return p_loss, q1_loss[0], q2_loss[0], v_loss[0], pi,logp_pi, grad_ys

    def policy_action(self,states, deterministic=False):
        mu,pi,logp_pi = self.actor.predict(states)
        action = mu if deterministic else pi 
        return action 

    def bellman_q_value(self,rewards, next_states, dones):
        q_target = np.zeros_like(rewards)
        if next_states is not None:
            future_rewards = self.target_vf.predict(next_states)
        else:
            future_rewards = np.zeros_like(rewards)

        for i in range(rewards.shape[0]):
            if dones[i]:
                q_target[i] = rewards[i]
            else:
                q_target[i] = rewards[i] + self.discount*future_rewards[i]
        return q_target

    def memorize(self,state, action, reward, done,new_state):
        if self.buffer.with_per:
            #(TODO): calculate td-error here 
            td_error=0 
        else 
            td_error=0
        state = state.reshape(-1)
        action = action.reshape(-1)
        self.buffer.memorize(state,action,reward, done,new_state, td_error)

    def sample_batch(self,batch_size):
        return self.buffer.sample_batch(batch_size)
    
    def train(self,args, summary_writer, train_data=None, val_data=None, test_data=None):
        max_val_rate=0
        val_data= np.asarray(val_data)
        tqdm_e=tqdm(range(args.batchs),desc='score',leave=True, unit='epoch')
        if train_data is None:
            dataset = CsvBuffer(arg)s.file_dir, args.reg_pattern, chunsize = args.batch_size)
            assert dataset.is_buffer_available,' neither train_data nor csv buffer is availble'
        else:
            dataset = Dataset(train_data,1,shuffle=True)
        warm_up = 20*args.batch_size

        for e in tqdm_e:
            batch_data = next(dataset)
            states,labels = batch_size[:,:-1],batch_size[:,-1].astype(int)
            a = self.policy_action(states,deterministic=False) #(batch, n_action)
            #llr = np.clip(np.log(a/(1-a)+1e-6),-5,5)
            r = np.where(labels=1,a.ravel(),-a.ravel())
            self.memorize(states,a,r,True,None)
            if e<warm_up:
                continue
            states, a,r,_,_,_ = self.sample_batch(args.batch_size)
            q_= self.bellman_q_value(rewards=r, next_states=None, dones=[True]*r.shape[0]) #(batch,)
            p_loss ,q1_loss,q2_loss,v_loss, tpi, tlogp_pi, tgrad_ys = self.update_model(states,a,q_.reshape(-1,1))

            score = r.mean()

            if e%self.assess_interval == 0 or e==args.batch_size-1 or e==warm_up:
                if val_data:
                    mu,pi, logp_pi = self.actor.predict(val_data[:,:-1])
                    val_pred = 1./(1.+np.exp(-mu))
                    val_y = val_data[:,-1]
                    auc = roc_auc_score(val_y, val_pred.reshape(-1))
                    self.logger.log_tabular('val_mu','{:.4f}+{:.4f}'.format(mu.mean(),mu.std()))
                    self.logger.log_tabular('val_pi','{:.4f}+{:.4f}'.format(pi.mean(),pi.std()))
                    self.logger.log_tabular('auc','{:.4f}'.format(auc))
                    max_val_rate = val_rate if val_rate>max_val_rate else max_val_rate

                if test_data:
                    mu,pi, logp_pi = self.actor.predict(test_data[:,:-1])
                    val_pred = 1./(1.+np.exp(-mu))
                    val_y = test_data[:,-1]
                    auc = roc_auc_score(val_y, val_pred.reshape(-1))
                    self.logger.log_tabular('test_mu','{:.4f}+{:.4f}'.format(mu.mean(),mu.std()))
                    self.logger.log_tabular('test_pi','{:.4f}+{:.4f}'.format(pi.mean(),pi.std()))
                    self.logger.log_tabular('test_auc','{:.4f}'.format(auc))
                summary_writer.add_summary(tf_summary(['mean_reward'],[score]),global_step=e)
                #sess = keras.backend.get_session()
            self.logger.log_tabular('ploss','{:.4f}'.format(p_loss))
            self.logger.log_tabular('a(s)','{:.4f}+{:.4f}'.format(tpi.mean(),tpi.std()))
            self.logger.log_tabular('tgrad_ys','{:.4f}+{:.4f}'.format(tgrad_ys.mean(), tgrad_ys.std()))
            self.logger.log_tabular('reward','{:.4f}'.format(reward))
            self.logger.dump_tabular()
            tqdm_e.set_description('score:'+'{:.4f}'.format(score))
            tqdm_e.set_postfix(max_val_rate='{:.4f}'.format(max_val_rate),val_rate='{:4.f}'.foramt(val_rate))
            tqdm_e.refresh()

if __name__ =='__main__':
    import sys 
    import os 
    import operator 
    from datetime import datetime
    parser = argparse.ArgumentParser(description='training_params')
    parser.add_argument('--file_dir',type=str, default='../data/clean',help='folder contains train_data')
    parser.add_argument('--reg_pattern',type=str, default=r'train_hnb.csv',help='pattern to filter files in folder')
    parser.add_argument('--batchs',type =int, default=300000,help='number of batchs to train')
    parser.add_argument('--batch_size',type=int, default=64)
    parser.add_argument('--type',type=str,default='sac')
    parser.add_argument('--model_path',type=str, default='../model/RL')
    args = parser.parse_args(sys.argv[1:])

    train_data = pd.read_pickle(train_file)
    val_data = pd.read_pickle(val_file)

    np.random.shuffle(train_data)
    n_state = train_data.shape[1] - 1
    n_action = 1
    checkpoint_queen = MinHeap(max_size=5, compare_key=operator.itemgetter(0))
    logger = Logger(output_dir='../assets',output_fname='sac_epoch_log')

    config = get_session_config(0.4, True,'0')
    sac = SAC(n_state,n_action, log_std_max=1,log_std_min=-10,alpha=0.1, 
            actor_lr=0.001, critic_lr=0.001, vf_lr=0.001, polyavk=0.995, discount=0.99,
            assess_interval=10, save_interval=5000, buffer_size=20000, 
            logger=logger, checkpoint_queen=checkpoint_queen)
    summary_writer = tf.summary.FileWriter('../assets',graph=tf.get_default_graph(), filename_suffix= 
                                            datetime.now().strftime('_%m%d%H%M')+'_sac')
    atexit.register(summary_writer.close)

    with tf.Session(config=config) as sess:
        keras.backend.set_session(sess)
        sess.run(tf.global_variables_initializer())
        sac.train(args,summary_writer, train_data=train_data, val_data=val_data.values)
    


    


















