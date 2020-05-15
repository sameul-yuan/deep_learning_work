# -*- encoding:utf-8 -*-
import os
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import he_normal,TruncatedNormal,glorot_normal
from nn_modules import dense_batchnorm_layer,dense_dropout_layer
K = keras.backend


class Actor(object):
    def __init__(self, n_state, n_action, gamma = 0.99, lr =0.001, tau = 0.01, **kwargs):
        self.n_action = n_action
        self.n_state = n_state
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.kwargs = kwargs
        self.model = self._build()
        self.target_model = self._build()  # copy.copy(self.model)
        self.optimizer = self._optimizer()
        # keras.utils.plot_model(self.model, '../assets/actor.png', True, True)

    def _build(self):
        with K.name_scope('actor'):
            input = keras.Input(shape=(self.n_state,),name='in')
            x = dense_batchnorm_layer(input, 64, activation='lrelu', kernel_initializer='he_normal',
                                  kernel_regularizer = keras.regularizers.l2(self.kwargs.get('l2_reg',0.01)),
                                  name='l1', batch_norm=False)
            x = dense_batchnorm_layer(x, 32, activation='lrelu', kernel_initializer='he_normal',   #TruncatedNormal()
                                  kernel_regularizer = keras.regularizers.l2(self.kwargs.get('l2_reg',0.01)),
                                  name='l2')
            x = dense_batchnorm_layer(x, 16, activation='lrelu', kernel_initializer='he_normal',
                                  kernel_regularizer = keras.regularizers.l2(self.kwargs.get('l2_reg',0.01)),
                                  name='l3')
            output = dense_batchnorm_layer(x, 1, activation='sigmoid', kernel_initializer='glorot_normal',
                                       kernel_regularizer = keras.regularizers.l2(self.kwargs.get('l2_reg',0.01)),
                                       name='output',batch_norm=True)

        return keras.Model(input, output)

    def _optimizer(self):
        " caculate and apply gradients of actor output w.r.t network params"
        action_ys = K.placeholder(shape=(None, self.n_action))

        # params_grad = tf.gradients(self.model.output, self.model.trainable_weights, -action_ys) # summarized over batch
        # grads = zip(params_grad, self.model.trainable_weights)        # batch_size = K.shape(action_ys)[0]
        # updates= [tf.train.GradientDescentOptimizer(self.lr).apply_gradients(grads)] # return  operation  # tf.train.GradientDescentOptimizer() AdamOptimizer

        loss = K.mean(-action_ys * self.model.output,axis=0)
        optimizer = keras.optimizers.Adam(self.lr)
        updates = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)

        updates += self.model.updates
        return K.function([self.model.input, action_ys], [self.model.output, loss], updates=updates)  # output should be a tensor, keras support output=[], but tf.keras not;

    def train(self, states, actions, grad_ys):
        actor_output =  self.optimizer([states,grad_ys])
        return actor_output # K.function return a list of array, only one output

    def predict(self, state):
        # return self.model.predict(np.expand_dims(state, axis=0))
        return self.model.predict(state)

    def target_predict(self, state):
        return self.target_model.predict(state)

    def copy_weights(self):
        w, target_w = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(w)):
            target_w[i] = self.tau * w[i] + (1 - self.tau)* target_w[i]
        self.target_model.set_weights(target_w)

    def save(self, path):
        self.model.save_weights(os.path.join(path,'actor_weights.h5'))

    def load_weights(self, file):
        self.model.load_weights(file)

    def view_layer(self,lay_name):
        return K.function([self.model.input],[self.model.get_layer(lay_name).output])
if __name__ == '__main__':
    # (x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()
    actor = Actor(200,1)

    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        K.set_session(sess)
        sess.run(tf.global_variables_initializer())
        states = np.random.randn(30, 200)
        grad_ys = np.random.randn(30,1)
        print(actor.view_layer('l1')(states)[0])
        actor.train(states, 0, grad_ys)
        print(actor.view_layer('l1')(states)[0])
        ret = actor.predict(states)
        print(ret)
        # K.learning_phase()
        # K.set_learning_phase()
