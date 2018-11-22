import tensorflow as tf
import math
import numpy as np

# Params of fully connected layers
FULLY_LAYER1_SIZE = 512
FULLY_LAYER2_SIZE = 512

# Params of conv layers
RECEPTIVE_FIELD1 = 8
RECEPTIVE_FIELD2 = 4
RECEPTIVE_FIELD3 = 4
# RECEPTIVE_FIELD4 = 3

STRIDE1 = 4
STRIDE2 = 2
STRIDE3 = 3
# STRIDE4 = 1

FILTER1 = 32
FILTER2 = 64
FILTER3 = 64
# FILTER4 = 64

# How fast is learning
LEARNING_RATE = 5e-4

# How much do we regularize the weights of the net
REGULARIZATION_DECAY = 0.0

# How fast does the target net track
TARGET_DECAY = 0.9999

# In what range are we initializing the weights in the final layer
FINAL_WEIGHT_INIT = 0.003

# How often do we plot variables during training
PLOT_STEP = 10


class CriticNetwork:

    def __init__(self, map_inputs, action_size, session, summary_writer):

        self.graph = session.graph

        with self.graph.as_default():

            # Get session and summary writer from ddpg
            self.sess = session
            self.summary_writer = summary_writer

            # Get input dimensions from ddpg
            self.action_size = action_size

            # Create critic network
            self.map_inputs = map_inputs
            self.action_input = tf.placeholder("float", [None, action_size], name="action_input")
            self.y_input = tf.placeholder("float", [None, 1], name="y_input")

            with tf.variable_scope('critic/network') as scope:
                self.Q_output = self.create_base_network()

                # Get all the variables in the critic network for exponential moving average, create ema op
                self.critic_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
                print(self.critic_variables)

            with tf.variable_scope('critic/target_network'):
                self.Q_output_target = self.create_base_network()

            net_vars        =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/network')
            target_net_vars =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/target_network')

                # L2 Regularization for all Variables
            with tf.name_scope('critic/regularization'):
                regularization_loss = tf.losses.get_regularization_loss(scope='critic/network')

            # Define the loss with regularization term
            with tf.name_scope('critic/cal_loss'):
                self.td_error = tf.reduce_mean(tf.pow(self.Q_output - self.y_input, 2))
                self.loss = self.td_error #+ regularization_loss
                q_value_summary             = tf.summary.scalar('q_value', tf.reduce_mean(self.Q_output))
                td_error_summary            = tf.summary.scalar('td_error', self.td_error)
                regularization_loss_summary = tf.summary.scalar('regularization_loss', regularization_loss)
                loss_summary                = tf.summary.scalar('loss', self.loss)

            # Define the optimizer
            with tf.name_scope('critic/q_param_opt'):
                self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

            # Define the action gradients for the actor training step
            with tf.name_scope('critic/q_gradients'):
                self.q_gradients = tf.gradients(self.Q_output, self.action_input)
                q_gradients_summary = []
                q_gradients_means = tf.reduce_mean(self.q_gradients[0], axis=0)

                for i in range(q_gradients_means.shape.as_list()[0]):
                    q_gradients_summary.append(tf.summary.scalar("q_gradient%d"%(i), q_gradients_means[i]))

            with tf.name_scope('critic/target_update'):
                with tf.name_scope('init_update'):
                    print("================================================================================================================================")
                    init_updates = []
                    for var, target_var in zip(net_vars, target_net_vars):
                        print("{} <- {}".format(target_var, var))
                        init_updates.append(tf.assign(target_var, var))
                    print("================================================================================================================================")

                    self.init_update = tf.group(*init_updates)

                with tf.name_scope('update'):
                    print("================================================================================================================================")
                    updates = []
                    for var, target_var in zip(net_vars, target_net_vars):
                        print("{} <- {}".format(target_var, var))
                        updates.append(tf.assign(target_var, TARGET_DECAY*target_var + (1 - TARGET_DECAY)*var))
                    print("================================================================================================================================")

                    with tf.control_dependencies([self.optimizer]):
                        self.update = tf.group(*updates)

            # Variables for plotting
            self.action_grads_mean_plot = [0, 0]
            self.td_error_plot = 0
            self.summary_merged = tf.summary.merge([q_value_summary, td_error_summary, regularization_loss_summary, loss_summary] + q_gradients_summary)

    def init(self):
        self.sess.run([self.init_update])

    def custom_initializer_for_conv(self):
        return tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_in', distribution='uniform', seed=None, dtype=tf.float32)

    def custom_initializer_for_dense(self):
        return tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_in', distribution='uniform', seed=None, dtype=tf.float32)

    def custom_initializer_for_final_dense(self):
        return tf.random_uniform_initializer(-FINAL_WEIGHT_INIT, FINAL_WEIGHT_INIT)

    def create_base_network(self):
        # new setup
        weight_decay = 1e-2

        outs = []
        for single_input in self.map_inputs:
            # conv layer1
            out = tf.layers.conv2d(inputs=single_input, filters=FILTER1, kernel_size=RECEPTIVE_FIELD1, strides=STRIDE1, padding='VALID',
                    data_format='channels_first',
                    kernel_initializer=self.custom_initializer_for_conv(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                    activation=tf.nn.relu)
            # conv layer2
            out = tf.layers.conv2d(inputs=out, filters=FILTER2, kernel_size=RECEPTIVE_FIELD2, strides=STRIDE2, padding='VALID',
                    data_format='channels_first',
                    kernel_initializer=self.custom_initializer_for_conv(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                    activation=tf.nn.relu)
            # conv layer3
            out = tf.layers.conv2d(inputs=out, filters=FILTER3, kernel_size=RECEPTIVE_FIELD3, strides=STRIDE3, padding='VALID',
                    data_format='channels_first',
                    kernel_initializer=self.custom_initializer_for_conv(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                    activation=tf.nn.relu)
            out = tf.layers.flatten(out)
            outs.append(out)

        # dense layer1
        out = tf.concat(outs, axis=1)
        out = tf.concat([out, self.action_input], axis=1)
        out = tf.layers.dense(inputs=out, units=FULLY_LAYER1_SIZE,
                kernel_initializer=self.custom_initializer_for_dense(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                bias_initializer=tf.zeros_initializer(),
                activation=tf.nn.relu)
        # dense layer2
        out = tf.layers.dense(inputs=out, units=FULLY_LAYER2_SIZE,
                kernel_initializer=self.custom_initializer_for_dense(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                bias_initializer=tf.zeros_initializer(),
                activation=tf.nn.relu)
        # dense layer3
        out = tf.layers.dense(inputs=out, units=1,
                kernel_initializer=self.custom_initializer_for_final_dense(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                bias_initializer=self.custom_initializer_for_final_dense(),
                activation=None)

        return out

    def train(self, training_step, y_batch, state_batch_list, action_batch):

        feed_dict={self.y_input: y_batch, self.action_input: action_batch,}
        for idx, single_batch in enumerate(state_batch_list):
            feed_dict[self.map_inputs[idx]] = single_batch

        # Run optimizer and compute some summary values
        if training_step%100 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            summary, td_error_value, _ = self.sess.run([self.summary_merged, self.td_error, self.update],
                    feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            self.summary_writer.add_run_metadata(run_metadata, 'c step%d' % training_step)
            self.summary_writer.add_summary(summary, training_step)
        else:
            td_error_value, _ = self.sess.run([self.td_error, self.update], feed_dict=feed_dict)


        # Increment the td error plot variable for td error average plotting
        self.td_error_plot += td_error_value

    def get_q_gradient(self, state_batch_list, action_batch):
        feed_dict={self.action_input: action_batch}
        for idx, single_batch in enumerate(state_batch_list):
            feed_dict[self.map_inputs[idx]] = single_batch

        # Get the action gradients for the actor optimization
        q_gradients = self.sess.run(self.q_gradients, feed_dict=feed_dict)[0]
        return q_gradients

    def evaluate(self, state_batch_list, action_batch):
        feed_dict={self.action_input: action_batch}
        for idx, single_batch in enumerate(state_batch_list):
            feed_dict[self.map_inputs[idx]] = single_batch

        return self.sess.run(self.Q_output, feed_dict=feed_dict)

    def target_evaluate(self, state_batch_list, action_batch):
        feed_dict={self.action_input: action_batch}
        for idx, single_batch in enumerate(state_batch_list):
            feed_dict[self.map_inputs[idx]] = single_batch

        return self.sess.run(self.Q_output_target, feed_dict=feed_dict)
