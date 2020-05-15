

import sys
import os
import argparse
import atexit
import warnings
import random

import numpy as np
from  tqdm import tqdm
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# sys.path.append(os.path.dirname(os.path.split(os.path.abspath(__file__))[0]))
from actor import Actor
from critic import  Critic
from dataset import  CsvBuffer, Dataset
# from utils.ou import OrnsteinUhlenbeckProcess
from metrics import get_session_config, tf_summary
from logger import  Logger, colorize
from min_heap import MinHeap
# from utils.nn_utils import  train_sampling

warnings.filterwarnings(action='ignore',category=FutureWarning)
# print(colorize(os.path.dirname(os.path.split(os.path.abspath(__file__))[0]),'blue',True))
# np.random.seed(2019)
# tf.set_random_seed(2019)
# random.seed(2019)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class DDPG(object):
    """deep deterministic policy gradient
    """
    def __init__(self, n_state, n_action, a_bound, gamma=0.99, tau=0.01, actor_lr=0.0005,critic_lr= 0.001,
                 noise_std=0.1, noise_decay=0.9995, noise_decay_steps=1000,
                 buffer_size=20000,save_interval= 5000, assess_interval= 10, logger=None, checkpoint_queen =None):
        self.logger = logger
        self.logger.save_config(locals())
        self.n_action = n_action
        self.n_state = n_state
        self.a_bound = a_bound
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        self.noise_decay_steps = noise_decay_steps
        self.pointer = 0
        self.buffer_size = buffer_size
        self.save_interval = save_interval
        self.assess_interval = assess_interval
        self.actor = Actor(self.n_state, self.n_action, gamma=gamma, lr=actor_lr, tau=tau, l2_reg=0)
        self.critic = Critic(self.n_state,self.n_action, gamma=gamma, lr=critic_lr, tau=tau,l2_reg=0)
        self.merge = self._merge_summary()
        self.ckpt_queen = checkpoint_queen

        self.prefix = self.__class__.__name__.lower()

    def _merge_summary(self):
        tf.summary.histogram('critic_output', self.critic.model.output)
        tf.summary.histogram('actor_output', self.actor.model.output)
        tf.summary.histogram('critic_dense1',self.critic.model.get_layer('l1').weights[0])
        tf.summary.histogram('actor_dense1',self.actor.model.get_layer('l1').weights[0])
        tf.summary.histogram('critic_dense2',self.critic.model.get_layer('l2').weights[0])
        tf.summary.histogram('actor_dense2',self.actor.model.get_layer('l2').weights[0])
        return tf.summary.merge_all()

    def policy_action(self,state):
        return self.actor.predict(state)

    def bellman_q_value(self, rewards, q_nexts, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        q_target = np.zeros_like(rewards) # asarry( copy = False), array(cope=True)
        for i in range(rewards.shape[0]):
            if dones[i]:
                q_target[i] = rewards[i]
            else:
                q_target[i] = rewards[i] + self.critic.gamma * q_nexts[i]
        return q_target

    def update_model(self, states, actions, q_values):
        # train critic
        loss_names,loss_values = self.critic.train_on_batch(states,actions, q_values)

        # train actor
        # p_actions = self.actor.predict(states)  #actions with no noise
        grad_ys = self.critic.gradients(states, self.actor.predict(states)) #(batch, n-action)
        actor_output = self.actor.train(states, self.actor.predict(states), grad_ys)

        # copy network
        self.actor.copy_weights()
        self.critic.copy_weights()

        # print(grad_ys, grad_ys.shape)
        # print(actor_output[0],actor_output[0].shape)
        # print(np.mean(grad_ys*actor_output[0]))

        return loss_names, loss_values, grad_ys, actor_output


    def save_weights(self,path):
        self.actor.save(path)
        self.critic.save(path)

    def save_model(self,path, file):
        self.actor.model.save(os.path.join(path, self.prefix+'_actor_'+file+'.h5'))
        self.critic.model.save(os.path.join(path, self.prefix+'_critic_'+file+'.h5'))

    def checkpoint(self, path, step,  metric_value):
        signature = str(step)+'_'+ '{:.4f}'.format(metric_value)
        to_delete, need_save = self.ckpt_queen.add((metric_value,signature))
        if to_delete:
            actor = os.path.join(path, self.prefix+'_actor_'+to_delete[1]+'.h5')
            critic = os.path.join(path,self.prefix+'_critic_'+to_delete[1]+'.h5')
            os.remove(actor)
            os.remove(critic)
        if need_save:
            self.save_model(path, signature)

    def train(self, args, summary_writer, train_data=None, val_data=None, test_data=None):
        results =[]
        max_val_rate =0
        val_data =np.asarray(val_data) # none will be array(None)
        # First, gather experience
        tqdm_e = tqdm(range(args.batchs), desc='score', leave=True, unit=" epoch")
        if train_data is None:
            dataset = CsvBuffer(args.file_dir, args.reg_pattern, chunksize=args.batch_size)   # 100*(20+1)
            assert  dataset.is_buffer_available, 'neither train_data nor csv buffer is available'
        # noise = OrnsteinUhlenbeckProcess(size=self.n_action)
        else:
            dataset = Dataset(train_data, args.batch_size, shuffle=True)
        for e in tqdm_e:
            batch_data = next(dataset)
            states, labels = batch_data[:,:-1], batch_data[:,-1].astype(int)

            a = self.policy_action(states)  #(batch, n_action)
            a = np.clip(a + np.random.normal(0, self.noise_std, size=a.shape), self.a_bound[0], self.a_bound[1])
            # a = np.clip(np.random.normal(a, self.noise_std), self.a_bound[0], self.a_bound[1])
            # a = np.clip(a + noise.generate(time, a.shape[0]), self.a_bound[0], self.a_bound[1])
            llr =np.clip(np.log(a/(1-a) + 1e-6),-5,5)
            r = np.where(labels==1, llr.ravel(), -llr.ravel())  #(batch,)
            # q_nexts = self.critic.target_predict(new_states, self.actor.target_predict(new_states))
            q_= self.bellman_q_value(rewards=r, q_nexts=0, dones=[True]*r.shape[0]) #(batch,)
            loss_names, loss_values, grad_ys, actor_output = self.update_model(states,a, q_.reshape(-1,1))

            score = r.mean()

            if ((e+1)%self.noise_decay_steps-1)==0:
                self.noise_std *= self.noise_decay
                self.logger.log_tabular('noise', self.noise_std)
            if e%self.assess_interval == 0 or e == args.batchs -1:
                if val_data is not None:
                    val_pred= self.actor.predict(val_data[:,:-1])
                    val_y = val_data[:,-1]
                    val_rate, top_k = top_ratio_hit_rate(val_y.ravel(), val_pred.ravel())
                    self.logger.log_tabular('val_rate',val_rate)
                    self.logger.log_tabular('val_k',int(top_k))
                    self.checkpoint(args.model_path, e, val_rate)
                    max_val_rate = val_rate if val_rate>max_val_rate else max_val_rate
                if test_data is not None:
                    test_pred = self.actor.predict(test_data[:,:-1])
                    test_y = test_data[:,-1]
                    test_rate, top_k = top_ratio_hit_rate(test_y,test_pred.ravel())
                    self.logger.log_tabular('test_rate',test_rate)
                    self.logger.log_tabular('test_k',int(top_k))


                summary_writer.add_summary(tf_summary(['mean-reward'],[score]), global_step=e)
                summary_writer.add_summary(tf_summary(loss_names,[loss_values]),global_step=e)
                merge = keras.backend.get_session().run(self.merge,feed_dict={
                    self.critic.model.input[0]:states,
                    self.critic.model.input[1]:a,
                    self.actor.model.input:states})
                summary_writer.add_summary(merge,global_step=e)

            for name, val in zip(loss_names,[loss_values]):
                self.logger.log_tabular(name, val)
            # print(grad_ys,grad_ys.shape)
            # print(actor_output)
            self.logger.log_tabular('dQ/da','%.4f+%.4f'%(grad_ys.mean(), grad_ys.std())) # grad_ys (batch,act_dim)
            self.logger.log_tabular('aout','%.4f+%.4f'%(actor_output[0].mean(), actor_output[0].std()))
            self.logger.log_tabular('aloss','%.4f'%(actor_output[1]))
            self.logger.log_tabular('reward', '%.4f+%.4f'%(score, r.std()))
            self.logger.dump_tabular()
            tqdm_e.set_description("score: " + '{:.4f}'.format(score))
            tqdm_e.set_postfix(noise_std='{:.4f}'.format(self.noise_std), max_val_rate='{:.4f}'.format(max_val_rate),val_rate='{:.4f}'.format(val_rate), top_k=top_k)
            tqdm_e.refresh()

        return results


if __name__ == '__main__':
    import sys
    import os
    import operator
    from datetime import datetime
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--file_dir', type=str, default='../data/clean', help="folder contains train data")
    parser.add_argument('--reg_pattern', type=str, default=r'train_hnb.csv', help="pattern to filter files in file_dir")
    parser.add_argument('--batchs', type=int, default=400000, help="number of batchs to train")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--type',type=str, default='ddpg')
    parser.add_argument('--model_path',type=str, default='../model/RL')
    args = parser.parse_args(sys.argv[1:])

    train_data = pd.read_pickle('../data/clean/train_bj_dl_107_woe_tail.pkl')
    val_data = pd.read_pickle('../data/clean/val_bj_dl_107_woe_tail.pkl')

    # demo_df  = pd.read_csv('../data/clean/train_hnb.csv',nrows=1)
    train_data = train_sampling(train_data, col='is_y2', method='down', pn_ratio=0.2,seed=2019)
    # train_data = train_sampling(train_data, col='is_y2', method='up', pn_ratio=0.5,seed=2019)
    pn_ratio = sum(train_data.is_y2==1)/sum(train_data.is_y2==0)

    print(train_data.head())
    print(val_data.head())
    train_data = train_data.values
    np.random.shuffle(train_data)
    # np.random.shuffle(train_data)
    n_state= train_data.shape[1]-1
    n_action = 1
    print(colorize('pn-ratio={}'.format(pn_ratio),'cyan',True))
    print(colorize('action_dim=%d, state_dim=%d'%(n_action,n_state), 'cyan',True))
    print(colorize('train_shape={}, val_shape={}'.format(train_data.shape, val_data.shape),'cyan',True))

    checkpoint_queen = MinHeap(max_size=5, compare_key=operator.itemgetter(0))
    logger = Logger(output_dir='../assets', output_fname='ddpg_epoch_log')

    config = get_session_config(frac=0.4, allow_growth=True, gpu="0")
    ddpg = DDPG(n_state=n_state, n_action=n_action, a_bound=[0.0001,0.9999], logger=logger, checkpoint_queen=checkpoint_queen,
                actor_lr=0.001, critic_lr=0.001)

    summary_writer =  tf.summary.FileWriter('../assets', graph= tf.get_default_graph(),filename_suffix = datetime.now().strftime('_%m%d'))
    atexit.register(summary_writer.close)
    # saver = tf.train.Saver(max_to_keep=5) # store_model

    with tf.Session(config=config) as sess:
        keras.backend.set_session(sess)
        sess.run(tf.global_variables_initializer())
        ddpg.train(args,summary_writer,train_data=train_data, val_data=val_data.values)










