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
LEARNING_RATE = 1e-4

# How much do we regularize the weights of the net
REGULARIZATION_DECAY = 0.0

# How fast does the target net track
TARGET_DECAY = 0.9999

# In what range are we initializing the weights in the final layer
FINAL_WEIGHT_INIT = 0.003

# How often do we plot variables during training
PLOT_STEP = 10

class QValueNetwork:

    def __init__(self, map_input, action_space, action_res, session, summary_writer, training_step_variable):
        self.graph = session.graph

        with self.graph.as_default():

            # Get session and summary writer from ddpg
            self.sess = session
            self.summary_writer = summary_writer

            # Get input dimensions from ddpg
            self.action_res = action_res
            self.action_space = action_space

            # Create q_value network
            self.map_input = map_input
            self.y_input = tf.placeholder("float", [None, 1], name="y_input")

            with tf.variable_scope('q_value/network') as scope:
                self.Q_output = self.create_base_network()

                # Get all the variables in the q_value network for exponential moving average, create ema op
                self.Q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
                print(self.Q_vars)

                # L2 Regularization for all Variables
            with tf.name_scope('q_value/regularization'):
                regularization_loss = tf.losses.get_regularization_loss(scope='q_value/network')

            # Define the loss with regularization term
            with tf.name_scope('q_value/cal_loss'):
                self.td_error = tf.reduce_mean(tf.pow(self.Q_output - self.y_input, 2))
                self.loss = self.td_error # + regularization_loss
                td_error_summary            = tf.summary.scalar('td_error', self.td_error)
                regularization_loss_summary = tf.summary.scalar('regularization_loss', regularization_loss)
                loss_summary                = tf.summary.scalar('loss', self.loss)

            # Define the optimizer
            with tf.name_scope('q_value/q_param_opt'):
                self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss, global_step=training_step_variable)

            with tf.variable_scope('q_value/target_network') as scope:
                self.Q_output_target = self.create_base_network()
                self.Q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
                print(self.Q_target_vars)
            with tf.variable_scope('q_value/target_network') as scope:
                target_updates = []
                for var, target_var in zip(self.Q_vars, self.Q_target_vars):
                    target_updates.append(target_var.assign(var))
                self.target_updates = tf.group(*target_updates)

            # Variables for plotting
            self.summary_merged = tf.summary.merge([td_error_summary, regularization_loss_summary, loss_summary])

    def custom_initializer_for_conv(self):
        return tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_in', distribution='uniform', seed=None, dtype=tf.float32)

    def custom_initializer_for_dense(self):
        return tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_in', distribution='uniform', seed=None, dtype=tf.float32)

    def custom_initializer_for_final_dense(self):
        return tf.random_uniform_initializer(-FINAL_WEIGHT_INIT, FINAL_WEIGHT_INIT)

    def create_base_network(self):
        # new setup
        weight_decay = 1e-2

        # conv layer1
        out = tf.layers.conv2d(inputs=self.map_input, filters=FILTER1, kernel_size=RECEPTIVE_FIELD1, strides=STRIDE1, padding='VALID',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                bias_initializer=tf.contrib.layers.xavier_initializer(),
                activation=tf.nn.relu)
        # conv layer2
        out = tf.layers.conv2d(inputs=out, filters=FILTER2, kernel_size=RECEPTIVE_FIELD2, strides=STRIDE2, padding='VALID',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                bias_initializer=tf.contrib.layers.xavier_initializer(),
                activation=tf.nn.relu)
        # conv layer3
        out = tf.layers.conv2d(inputs=out, filters=FILTER3, kernel_size=RECEPTIVE_FIELD3, strides=STRIDE3, padding='VALID',
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                bias_initializer=tf.contrib.layers.xavier_initializer(),
                activation=tf.nn.relu)

        # dense layer1
        size = 1
        for n in out.get_shape().as_list()[1:]:
            size *= n
        out = tf.reshape(out, [-1, size])
        #out = tf.concat([out, self.action_input], axis=1)
        out = tf.layers.dense(inputs=out, units=FULLY_LAYER1_SIZE,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                #bias_initializer=tf.zeros_initializer(),
                bias_initializer=tf.contrib.layers.xavier_initializer(),
                activation=tf.nn.relu)
        # dense layer2
        out = tf.layers.dense(inputs=out, units=FULLY_LAYER2_SIZE,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                #bias_initializer=tf.zeros_initializer(),
                bias_initializer=tf.contrib.layers.xavier_initializer(),
                activation=tf.nn.relu)
        # dense layer3
        out = tf.layers.dense(inputs=out, units=self.action_space,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                #bias_initializer=tf.zeros_initializer(),
                bias_initializer=tf.contrib.layers.xavier_initializer(),
                activation=None)

        return out

    def target_update(self):
        _ = self.sess.run([self.target_updates])

    def train(self, training_step, y_batch, state_batch):

        # Run optimizer and compute some summary values
        if training_step%100 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            summary, td_error_value, _ = self.sess.run([self.summary_merged, self.td_error, self.optimizer],
                    feed_dict={
                                self.y_input: y_batch,
                                self.map_input: state_batch,
                                },
                    options=run_options, run_metadata=run_metadata)
            self.summary_writer.add_run_metadata(run_metadata, 'c step%d' % training_step)
            self.summary_writer.add_summary(summary, training_step)
        else:
            td_error_value, _ = self.sess.run([self.td_error, self.optimizer],
                                                feed_dict={
                                                    self.y_input: y_batch,
                                                    self.map_input: state_batch,
                                                    })

    def evaluate(self, state_batch):
        return self.sess.run(self.Q_output, feed_dict={self.map_input: state_batch})

    def target_evaluate(self, state_batch):
        return self.sess.run(self.Q_output_target, feed_dict={self.map_input: state_batch})

    def get_action(self, state):
        Q_output_target_value = self.sess.run(self.Q_output_target, feed_dict={self.map_input: [state]})[0]
        print('self.Q_output_target')
        print(self.Q_output_target)
        print('Q_output_target_value')
        print(Q_output_target_value.reshape(self.action_res))
        action_id = np.argmax(Q_output_target_value)
        print('action_id')
        print(action_id)
        return action_id
