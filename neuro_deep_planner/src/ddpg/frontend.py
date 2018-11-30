import tensorflow as tf
import numpy as np

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

# How fast does the target net track
TARGET_DECAY = 0.9999

class FrontEndNetwork:

    def __init__(self, name, map_inputs, session, summary_writer, training_step_variable):

        self.graph = session.graph
        self.name = name

        with self.graph.as_default():

            # Get session and summary writer from ddpg
            self.sess = session
            self.summary_writer = summary_writer

            # Create frontend network
            self.map_inputs = map_inputs
            #tf.summary.scalar('map_inputs', tf.reduce_mean(self.map_inputs[0]))

            with tf.variable_scope('{}/network'.format(self.name)) as scope:
                self.output     = self.create_base_network(self.map_inputs)
                self.net_vars   =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

            with tf.variable_scope('{}/target_network'.format(self.name)) as scope:
                self.output_target   = self.create_base_network(self.map_inputs)
                self.target_net_vars =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

            with tf.name_scope('{}/regularization'.format(self.name)):
                self.regularization_loss = tf.losses.get_regularization_loss(scope='{}/network'.format(name))
                regularization_loss_summary = tf.summary.scalar('regularization_loss', self.regularization_loss)

            with tf.name_scope('{}/target_update/'.format(self.name)):
                with tf.name_scope('init_update'):
                    print("================================================================================================================================")
                    init_updates = []
                    for var, target_var in zip(self.net_vars, self.target_net_vars):
                        print("{} <- {}".format(target_var, var))
                        init_updates.append(tf.assign(target_var, var))
                    print("================================================================================================================================")

                    self.init_update = tf.group(*init_updates)

            # Variables for plotting
            self.summary_merged = tf.summary.merge([regularization_loss_summary])

    def set_update(self):
        with tf.name_scope('{}/target_update/update'.format(self.name)):
            print("================================================================================================================================")
            updates = []
            for var, target_var in zip(self.net_vars, self.target_net_vars):
                print("{} <- {}".format(target_var, var))
                update = tf.assign(target_var, TARGET_DECAY*target_var + (1 - TARGET_DECAY)*var)
                updates.append(update)
            print("================================================================================================================================")

            self.update = tf.group(*updates)
            return self.update

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
            # conv layer4
#            out = tf.layers.conv2d(inputs=out, filters=FILTER4, kernel_size=RECEPTIVE_FIELD4, strides=STRIDE4, padding='VALID',
#                    data_format='channels_first',
#                    kernel_initializer=self.custom_initializer_for_conv(),
#                    kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
#                    activation=tf.nn.relu)
            out = tf.layers.flatten(out)
            outs.append(out)

        out = tf.concat(outs, axis=1)
        return out

