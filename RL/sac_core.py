import os 
import sys
import numpy as np 
from tensorflow import keras 
# sys.path.append(os.path.dirname(os.path.split(os.path.abspath(__file__))[0]))
from nn_modules import dense_batchnorm_layer, dense_dropout_layer

K = keras.backend

class GaussianPolicy(object):
    def __init__(self, n_state, n_action, a_scale=5, log_std_max=2, log_std_min=-20,**kwargs):
        self.n_state =n_state
        self.n_action = n_action
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min 
        self.kwargs = kwargs 
        self.a_scale = a_scale
    @staticmethod
    def _gaussain_likelihood(x,mu,log_std):
        # return log(p) with p ~ N(mu,std)
        pre_sum = -0.5*(((x-mu)/(K.exp(log_std)+K.epsilon()))**2+ 2*log_std + np.log(2*np.pi))
        return K.sum(pre_sum, axis=1,keepdims=True)
    
    @staticmethod
    def _clip_but_pass_gradient(x,l=-1.,u=1.):
        """
        clip x to (l,u)
        """
        clip_up = K.cast(x>u, K.floatx())
        clip_low = K.cast(x<l,K.floatx())
        return x+K.stop_gradient((u-x)*clip_up + (l-x)*clip_low)
    
    @staticmethod
    def _appply_squash_function(mu,pi, logp_pi):
        #apply invertible squash function 'tanh' to gausian output
        mu = K.tanh(mu)
        pi = K.tanh(pi)
        logp_pi -= K.sum(K.log(__class__._clip_but_pass_gradient(1-pi**2,l=0,u=1.)+K.epsilon()),axis=1,keepdims=True)
        return [mu,pi,logp_pi]
    
    def build_actor(self,activation='relu',hidden_sizes=(64,32,16),name_scope=None ):
        n_state = self.n_state
        n_action = self.n_action

        with K.name_scope(name_scope) as scope:
            input = keras.Input(shape=(n_state,),name='states')
            batch_norms = [False] +[True]*(len(hidden_sizes)-1)
            x = input 
            for i, hidden in enumerate(hidden_sizes):
                x = dense_batchnorm_layer(x,hidden, activation=activation, kernel_initializer='he_normal', #TruncateNormal()
                    kernel_regularizer=keras.regularizers.l2(self.kwargs.get('l2_reg',0.01)),name='L'+str(i),batch_norm=batch_norms[i])
            mu = dense_batchnorm_layer(x,n_action, activation=None, kernel_initializer='glorot_normal',
                    kernel_regularizer=keras.regularizers.l2(self.kwargs.get('l2_reg',0.01)),name='mu',batch_norm=False)
            log_std = dense_batchnorm_layer(x,n_action, activation='tanh', kernel_initializer='glorot_normal',
                    kernel_regularizer=keras.regularizers.l2(self.kwargs.get('l2_reg',0.01)),name='logstd',batch_norm=False)
            #clip 
            log_std = keras.layers.Lambda(lambda x: self.log_std_min + 0.5*(x+1)*(self.log_std_max-self.log_std_min))(log_std)
            std  = keras.layers.Lambda(lambda x: K.exp(x))(log_std)
            #a = mu + N(0,1)*std
            pi = keras.layers.Lambda(lambda x: x + K.random_normal(K.shape(x))*std,output_shape=lambda shape:shape)(mu)
            #log(p(a))
            logp_pi = keras.layers.Lambda(lambda x: self._gaussain_likelihood(x[0],x[1],x[2]),output_shape=lambda x_shapes:x_shapes[0])([pi,mu,log_std])
            # mu~(-1,1),pi~(-1,1)
            mu,pi,logp_pi = keras.layers.Lambda(lambda x:self._appply_squash_function(x[0],x[1],x[2]), 
                                  output_shape=lambda x_shapes:[x_shapes[0],x_shapes[1],x_shapes[2]])([mu,pi,logp_pi])
            # mu ~(-a_scale, a_scale)
            mu = keras.layers.Lambda(lambda x: x*self.a_scale,output_shape=lambda shape:shape)(mu)
            pi = keras.layers.Lambda(lambda x: x*self.a_scale,output_shape=lambda shape:shape)(pi)
        return keras.Model(inputs=input, outputs=[mu, pi,logp_pi])

    def build_critic(self,activation='relu',hidden_sizes=(64,32),name_scope=None):
        n_state = self.n_state
        n_action = self.n_action
        with K.name_scope(name_scope) as scope:
            s_input = keras.Input(shape=(n_state,),name='c_states')
            a_input = keras.Input(shape=(n_action,),name='c_action')
            input = keras.layers.concatenate([s_input,a_input],axis=1)
            batch_norms = [False] +[True]*(len(hidden_sizes)-1)
            x = input 
            for i, hidden in enumerate(hidden_sizes):
                x = dense_batchnorm_layer(x,hidden, activation=activation, kernel_initializer='he_normal',
                                            kernel_regularizer=keras.regularizers.l2(self.kwargs.get('l2_reg',0.01)),
                                            name='L'+str(i),batch_norm=batch_norms[i])
            q = dense_batchnorm_layer(x,1,activation='linear', kernel_initializer='truncated_normal',
                                        kernel_regularizer=keras.regularizers.l2(self.kwargs.get('l2_reg',0.01)),
                                        name='output',batch_norm=False)
        return keras.Model(inputs=[s_input,a_input],outputs=q)

    def build_vf(self, activation='relu',hidden_sizes=(64,32),name_scope=None):
        n_state = self.n_state
        with K.name_scope(name_scope,'vf') as scope:
            input = keras.Input(shape=(n_state,),name='states')
            batch_norms = [False] +[True]*(len(hidden_sizes)-1)
            x = input 
            for i,hidden in enumerate(hidden_sizes):
                x = dense_batchnorm_layer(x,hidden, activation=activation, kernel_initializer='he_normal', 
                                       kernel_regularizer=keras.regularizers.l2(self.kwargs.get('l2_reg',0.01)),
                                       name='L'+str(i),batch_norm=batch_norms[i])
            vf = dense_batchnorm_layer(x,1,activation=activation, kernel_initializer='truncated_normal',
                                    kernel_regularizer=keras.regularizers.l2(self.kwargs.get('l2_reg',0.01)),
                                    name='output',batch_norm=False)
        return keras.Model(inputs=input,outputs=vf)    

if __name__=='__main__':
    import numpy as np 
    policy = GaussianPolicy(100,1)
    actor = policy.build_actor()
    states = K.placeholder(shape=(None, 100))
    states = np.random.randn(10,100)
    mu,pi,logp_pi = actor(states)
    print(mu,pi,logp_pi)
    









    