import numpy as np
import tensorflow.compat.v1 as tf
import tflearn

FEATURE_NUM = 64
ACTION_EPS = 1e-4
GAMMA = 0.93
# PPO2
EPS = 0.2
PPO_TRAINING_EPO = 5

class Network():

    def CreateNetwork(self, inputs):
        with tf.variable_scope('actor'):
            split_arr = []
            for p in range(0, self.s_dim[0] - 3):
                net = tflearn.conv_1d(inputs[:, p:p+1, :], FEATURE_NUM, 4, activation='relu')
                split_arr.append(tflearn.flatten(net))
            for p in range(self.s_dim[0] - 3,self.s_dim[0]):
                net = tflearn.fully_connected(inputs[:, p:p+1, -1], FEATURE_NUM, activation='relu')
                split_arr.append(net)
            merge_net = tflearn.merge(split_arr, 'concat')
            # split_arr.append(net)
            # merge_net = tflearn.merge(split_arr, 'concat')
            
            pi_net = tflearn.fully_connected(merge_net, FEATURE_NUM, activation='relu')
            pi = tflearn.fully_connected(pi_net, self.a_dim, activation='tanh')

        return pi
            
    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    def load_model(self, nn_model):
        saver = tf.train.Saver(max_to_keep=1000)  # save neural net parameters
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(self.sess, nn_model)
            print("Model restored.")
            
    def __init__(self, state_dim, action_dim, learning_rate):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self._entropy = 0.
        
        self.lr_rate = learning_rate
        
        config=tf.ConfigProto(allow_soft_placement=True,
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        self.pi = self.CreateNetwork(inputs=self.inputs)
        
        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))
        
        self.loss = tflearn.mean_square(self.pi, self.acts)
        self.opt = tf.train.AdamOptimizer(self.lr_rate).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

    def predict(self, input):
        input_ = np.reshape(input, [-1, self.s_dim[0], self.s_dim[1]])
        action = self.sess.run(self.pi, feed_dict={
            self.inputs: input_
        })
        return action[0]
    
    def train(self, s_batch, a_batch):
        self.sess.run(self.opt, feed_dict={
            self.inputs: s_batch,
            self.acts: a_batch
        })

    def save_model(self, nn_model='model'):
        saver = tf.train.Saver(max_to_keep=1000)  # save neural net parameters
        if nn_model is not None:  # nn_model is the path to file
            saver.save(self.sess, nn_model)
            print("Model saved.")
