#! -*- encoding:utf-8 -*-

import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.initializers import he_normal,TruncatedNormal,glorot_normal
from nn_modules import  dense_batchnorm_layer,dense_dropout_layer
K = keras.backend

class Critic(object):
    def __init__(self, n_state, n_action, gamma = 0.99, lr =0.001, tau = 0.01,**kwargs):
        self.n_action = n_action
        self.n_state = n_state
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.kwargs = kwargs
        self.model = self._build()
        self.target_model = self._build()
        self._optimize()
        # keras.utils.plot_model(self.model, '../assets/critic.png', True, True)

    def _build(self):
        with K.name_scope('critic'):
            s_input = keras.Input(shape=(self.n_state,),name='sin')
            a_input = keras.Input(shape=(self.n_action,),name='ain')
            input = keras.layers.concatenate([s_input,a_input],axis=-1)
            x = dense_batchnorm_layer(input, 64, activation='relu', kernel_initializer='he_normal', # TruncatedNormal()
                                  kernel_regularizer = keras.regularizers.l2(self.kwargs.get('l2_reg',0.01)),
                                  name='l1', batch_norm=False)
            x = dense_batchnorm_layer(x, 32, activation='relu', kernel_initializer='he_normal',
                                      kernel_regularizer = keras.regularizers.l2(self.kwargs.get('l2_reg',0.01)),name='l2')
            x = dense_batchnorm_layer(x, 16, activation='relu', kernel_initializer=TruncatedNormal(),
                                      kernel_regularizer = keras.regularizers.l2(self.kwargs.get('l2_reg',0.01)),name='l3')

            output = dense_batchnorm_layer(x, 1, activation='linear', kernel_initializer='truncated_normal',
                                           kernel_regularizer = keras.regularizers.l2(self.kwargs.get('l2_reg',0.01)),
                                           name='output', batch_norm=False)
            # output = keras.layers.Lambda(lambda x: x*5)(output)

        return keras.Model(inputs=[s_input, a_input], outputs =output)

    def _optimize(self):
        # calculate gradient of critic w.r.t. action
        # self.action_grads = K.function([self.model.input[0], self.model.input[1]],K.gradients(K.mean(self.model.output,axis=0), [self.model.input[1]]))
        self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(self.model.output, [self.model.input[1]])) # (batch, len(xs))
        self.model.compile(keras.optimizers.Adam(self.lr), loss='mse',metrics=None)

    def gradients(self,states, actions):
        return self.action_grads([states, actions])[0]

    def train_on_batch(self,states,actions, targets):
        losses = self.model.train_on_batch([states,actions], targets)
        return self.model.metrics_names, losses

    def target_predict(self,states, actions):
        return self.target_model.predict([states, actions])

    def copy_weights(self):
        w, target_w = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(w)):
            target_w[i] = self.tau * w[i] + (1 - self.tau)* target_w[i]
        self.target_model.set_weights(target_w)

    def save(self, path):
        self.model.save_weights(os.path.join(path, 'critic_weights.h5'))

    def load_weights(self, file):
        self.model.load_weights(file)


if __name__ == '__main__':
    import  sys
    critic = Critic(100,1)
    states = np.random.randn(30,100)
    action = np.random.randn(30,1)
    target = np.random.randn(30,1)

    critic.train_on_batch(states,action,target)
    grads = critic.gradients(states,action)

    print(type(grads), grads.shape)


