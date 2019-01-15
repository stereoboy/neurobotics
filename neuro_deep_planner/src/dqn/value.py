import tensorflow as tf
import math
import numpy as np

# How fast is learning
LEARNING_RATE = 5e-5

# How much do we regularize the weights of the net
REGULARIZATION_DECAY = 0.0

# Weight decay for regularization
BASE_WEIGHT_DECAY = 1e-4

# In what range are we initializing the weights in the final layer
FINAL_WEIGHT_INIT = 0.003

# How often do we plot variables during training
PLOT_STEP = 10

# Target Network Update Step
TARGET_UPDATE_STEP = 3000

class QValueNetwork:

    def __init__(self, map_inputs, layers, action_space, action_res, session, summary_writer, training_step_variable):
        self.graph = session.graph
        self.layers = layers

        with self.graph.as_default():

            # Get session and summary writer from ddpg
            self.sess = session
            self.summary_writer = summary_writer

            # Get input dimensions from ddpg
            self.action_res = action_res
            self.action_space = action_space

            # Create q_value network
            self.map_inputs = map_inputs
            self.y_batch = tf.placeholder("float", [None, 1], name="y_batch")

            with tf.variable_scope('q_value/network') as scope:
                self.Q_output = self.create_base_network(self.map_inputs)

                # Get all the variables in the q_value network for exponential moving average, create ema op
                self.Q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

                # L2 Regularization for all Variables
            with tf.name_scope('q_value/regularization'):
                regularization_loss = tf.losses.get_regularization_loss(scope='q_value/network')

            # Define the loss with regularization term
            with tf.name_scope('q_value/cal_loss'):
                self.td_error = tf.reduce_mean(tf.pow(self.Q_output - self.y_batch, 2))
                self.loss = self.td_error # + regularization_loss
            with tf.name_scope('critic/cal_loss'):
                q_value_summary             = tf.summary.scalar('q_value', tf.reduce_mean(self.Q_output))
                td_error_summary            = tf.summary.scalar('td_error', self.td_error)
                regularization_loss_summary = tf.summary.scalar('regularization_loss', regularization_loss)
                loss_summary                = tf.summary.scalar('loss', self.loss)

            # Define the optimizer
            with tf.name_scope('q_value/q_param_opt'):
                self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss, global_step=training_step_variable)

            with tf.variable_scope('q_value/target_network') as scope:
                self.Q_output_target = self.create_base_network(self.map_inputs)
                self.Q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
            with tf.variable_scope('q_value/target_network') as scope:
                print("================================================================================================================================")
                target_updates = []
                for var, target_var in zip(self.Q_vars, self.Q_target_vars):
                    print("{} <- {}".format(target_var, var))
                    target_updates.append(target_var.assign(var))
                self.target_updates = tf.group(*target_updates)
                print("================================================================================================================================")

            # Variables for plotting
            self.summary_merged = tf.summary.merge([q_value_summary, td_error_summary, regularization_loss_summary, loss_summary])

    def custom_initializer_for_conv(self):
        return tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_in', distribution='uniform', seed=None, dtype=tf.float32)

    def custom_initializer_for_dense(self):
        return tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_in', distribution='uniform', seed=None, dtype=tf.float32)

    def custom_initializer_for_final_dense(self):
        return tf.random_uniform_initializer(-FINAL_WEIGHT_INIT, FINAL_WEIGHT_INIT)

    def create_base_network(self, input_list):
        # new setup
        weight_decay = 1e-2

        outs = []
        for single_input in input_list:
            out = single_input
            for kernel_size, strides, filter_size in self.layers['frontend']:
                out = tf.layers.conv2d(inputs=out, filters=filter_size, kernel_size=kernel_size, strides=strides, padding='VALID',
                        data_format='channels_first',
                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(BASE_WEIGHT_DECAY),
                        bias_initializer=tf.zeros_initializer(),
                        activation=tf.nn.relu)
            out = tf.layers.flatten(out)
            outs.append(out)

        out = tf.concat(outs, axis=1)

        # dense layer
        for layer_size in self.layers['backend']:
            out = tf.layers.dense(inputs=out, units=layer_size,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(BASE_WEIGHT_DECAY),
                    bias_initializer=tf.zeros_initializer(),
                    activation=tf.nn.relu)
        # final layer
        out = tf.layers.dense(inputs=out, units=self.action_space,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                bias_initializer=tf.zeros_initializer(),
                activation=None)

        return out

    def target_update(self):
        _ = self.sess.run([self.target_updates])

    def train(self, training_step, y_batch, state_batch_list):

        feed_dict={self.y_batch: y_batch}
        for idx, single_batch in enumerate(state_batch_list):
            feed_dict[self.map_inputs[idx]] = single_batch

        # Run optimizer and compute some summary values
        if training_step%100 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            summary, td_error_value, _ = self.sess.run([self.summary_merged, self.td_error, self.optimizer],
                    feed_dict=feed_dict,
                    options=run_options, run_metadata=run_metadata)
            self.summary_writer.add_run_metadata(run_metadata, 'c step%d' % training_step)
            self.summary_writer.add_summary(summary, training_step)
        else:
            td_error_value, _ = self.sess.run([self.td_error, self.optimizer],
                                                feed_dict=feed_dict)

        if training_step%TARGET_UPDATE_STEP:
            self.q_value_network.target_update()

    def evaluate(self, state_batch_list):
        feed_dict = {}
        for idx, single_batch in enumerate(state_batch_list):
            feed_dict[self.map_inputs[idx]] = single_batch
        return self.sess.run(self.Q_output, feed_dict=feed_dict)

    def target_evaluate(self, state_batch_list):
        feed_dict = {}
        for idx, single_batch in enumerate(state_batch_list):
            feed_dict[self.map_inputs[idx]] = single_batch
        return self.sess.run(self.Q_output_target, feed_dict=feed_dict)

    def get_action(self, state):
        feed_dict = {}
        for idx, single_state in enumerate(state):
            feed_dict[self.map_inputs[idx]] = [single_state]
        Q_output_target_value = self.sess.run(self.Q_output_target, feed_dict=feed_dict)[0]
        print('self.Q_output_target')
        print(self.Q_output_target)
        print('Q_output_target_value')
        print(Q_output_target_value.reshape(self.action_res))
        action_id = np.argmax(Q_output_target_value)
        print('action_id')
        print(action_id)
        return action_id
