import tensorflow as tf
import numpy as np

# Params of fully connected layers
FULLY_LAYER1_SIZE = 200
FULLY_LAYER2_SIZE = 200

# Params of conv layers
RECEPTIVE_FIELD1 = 4
RECEPTIVE_FIELD2 = 4
RECEPTIVE_FIELD3 = 4
# RECEPTIVE_FIELD4 = 3

STRIDE1 = 2
STRIDE2 = 2
STRIDE3 = 2
# STRIDE4 = 1

FILTER1 = 32
FILTER2 = 32
FILTER3 = 32
# FILTER4 = 64

# How fast is learning
LEARNING_RATE = 5e-4

# How fast does the target net track
TARGET_DECAY = 0.9999

# In what range are we initializing the weights in the final layer
FINAL_WEIGHT_INIT = 0.003

# How often do we plot variables during training
PLOT_STEP = 10


class FrontEndNetwork:

    def __init__(self, map_inputs, session, summary_writer, training_step_variable):

        self.graph = session.graph

        with self.graph.as_default():

            # Get session and summary writer from ddpg
            self.sess = session
            self.summary_writer = summary_writer

            # Create frontend network
            self.map_inputs = map_inputs
            #tf.summary.scalar('map_inputs', tf.reduce_mean(self.map_inputs[0]))

            with tf.variable_scope('frontend/network') as scope:
                self.output = self.create_base_network(self.map_inputs)
                self.net_vars        =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

            with tf.variable_scope('frontend/target_network') as scope:
                self.output_target = self.create_base_network(self.map_inputs)
                self.target_net_vars =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

            with tf.name_scope('frontend/regularization'):
                regularization_loss = tf.losses.get_regularization_loss(scope='frontend/network')
                regularization_loss_summary = tf.summary.scalar('regularization_loss', regularization_loss)

            with tf.name_scope('frontend/target_update'):
                with tf.name_scope('init_update'):
                    print("================================================================================================================================")
                    init_updates = []
                    for var, target_var in zip(self.net_vars, self.target_net_vars):
                        print("{} <- {}".format(target_var, var))
                        init_updates.append(tf.assign(target_var, var))
                    print("================================================================================================================================")

                    self.init_update = tf.group(*init_updates)

                with tf.name_scope('update'):
                    print("================================================================================================================================")
                    updates = []
                    for var, target_var in zip(self.net_vars, self.target_net_vars):
                        print("{} <- {}".format(target_var, var))
                        updates.append(tf.assign(target_var, TARGET_DECAY*target_var + (1 - TARGET_DECAY)*var))
                    print("================================================================================================================================")

                    self.update = tf.group(*updates)

            # Variables for plotting
            self.summary_merged = tf.summary.merge([regularization_loss_summary])

    def init(self):
        self.sess.run([self.init_update])

    def custom_initializer_for_conv(self):
        return tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_in', distribution='uniform', seed=None, dtype=tf.float32)

    def create_base_network(self, input_list):
        # new setup
        weight_decay = 1e-2

        outs = []
        for single_input in input_list:
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

        out = tf.concat(outs, axis=1)
        return out

#    def train(self, training_step, q_gradient_batch, state_batch_list):
#
#        feed_dict={self.q_gradient_input: q_gradient_batch}
#        for idx, single_batch in enumerate(state_batch_list):
#            feed_dict[self.map_inputs[idx]] = single_batch
#
#        # Train the frontend net
#        if training_step%100 == 0:
#            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#            run_metadata = tf.RunMetadata()
#            summary, _ = self.sess.run([self.summary_merged, self.update], feed_dict=feed_dict,
#                                                                                options=run_options, run_metadata=run_metadata)
#            self.summary_writer.add_run_metadata(run_metadata, 'a step%d' % training_step)
#            self.summary_writer.add_summary(summary, training_step)
#        else:
#            self.sess.run(self.update, feed_dict=feed_dict)
#
#    def get_action(self, state):
#        feed_dict = {}
#        for idx, single_state in enumerate(state):
#            feed_dict[self.map_inputs[idx]] = [single_state]
#        return self.sess.run(self.action_output, feed_dict=feed_dict)[0]
#
#    def evaluate(self, state_batch_list):
#        feed_dict = {}
#        for idx, single_batch in enumerate(state_batch_list):
#            feed_dict[self.map_inputs[idx]] = single_batch
#        # Get an action batch
#        actions = self.sess.run(self.action_output, feed_dict=feed_dict)
#        return actions
#
#    def target_evaluate(self, state_batch_list):
#        feed_dict = {}
#        for idx, single_batch in enumerate(state_batch_list):
#            feed_dict[self.map_inputs[idx]] = single_batch
#        # Get action batch
#        actions = self.sess.run(self.action_output_target, feed_dict=feed_dict)
#        return actions
