import tensorflow as tf
import numpy as np
from frontend import FrontEndNetwork

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


class ActorNetwork:

    def __init__(self, frontend, action_size, session, summary_writer, training_step_variable):

        self.graph = session.graph

        with self.graph.as_default():

            # Get session and summary writer from ddpg
            self.sess = session
            self.summary_writer = summary_writer

            # Get input dimensions from ddpg
            self.action_size = action_size

            # Create actor network
            self.frontend = frontend
            #tf.summary.scalar('map_inputs', tf.reduce_mean(self.map_inputs[0]))
            self.q_gradient_input = tf.placeholder("float", [None, action_size], name='q_gradient_input')

            with tf.variable_scope('actor/network') as scope:
                self.action_output = self.create_base_network(self.frontend.output)
                self.net_vars        =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

            with tf.variable_scope('actor/target_network') as scope:
                self.action_output_target = self.create_base_network(self.frontend.output_target)
                self.target_net_vars =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)

            self.actor_variables = self.net_vars + self.frontend.net_vars

            with tf.name_scope('actor/regularization'):
                regularization_loss = tf.losses.get_regularization_loss(scope='actor/network')
                regularization_loss_summary = tf.summary.scalar('regularization_loss', regularization_loss)

            # Define the gradient operation that delivers the gradients with the action gradient from the critic
            with tf.name_scope('actor/a_gradients'):
                self.parameters_gradients = tf.gradients(self.action_output, self.actor_variables, -self.q_gradient_input) #+ tf.gradients(regularization_loss, self.actor_variables, name="regularization")
                actor_gradient_summary = tf.summary.scalar('actor_gradients', tf.reduce_mean(self.parameters_gradients[0]))
            # Define the optimizer
            with tf.name_scope('actor/a_param_opt'):
                self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,
                                                                                       self.actor_variables), global_step=training_step_variable)

            with tf.name_scope('actor/target_update'):
                with tf.name_scope('init_update'):
                    print("================================================================================================================================")
                    init_updates = []
                    for var, target_var in zip(self.net_vars, self.target_net_vars):
                        print("{} <- {}".format(target_var, var))
                        init_updates.append(tf.assign(target_var, var))
                    print("================================================================================================================================")

                    with tf.control_dependencies([self.frontend.init_update]):
                        self.init_update = tf.group(*init_updates)

                with tf.name_scope('update'):
                    print("================================================================================================================================")
                    updates = []
                    for var, target_var in zip(self.net_vars, self.target_net_vars):
                        print("{} <- {}".format(target_var, var))
                        updates.append(tf.assign(target_var, TARGET_DECAY*target_var + (1 - TARGET_DECAY)*var))
                    print("================================================================================================================================")

                    with tf.control_dependencies([self.frontend.update, self.optimizer]):
                        self.update = tf.group(*updates)

            # Variables for plotting
            self.summary_merged = tf.summary.merge([actor_gradient_summary, regularization_loss_summary])

    def init(self):
        self.sess.run([self.init_update])

    def custom_initializer_for_conv(self):
        return tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_in', distribution='uniform', seed=None, dtype=tf.float32)

    def custom_initializer_for_dense(self):
        return tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_in', distribution='uniform', seed=None, dtype=tf.float32)

    def custom_initializer_for_final_dense(self):
        return tf.random_uniform_initializer(-FINAL_WEIGHT_INIT, FINAL_WEIGHT_INIT)

    def custom_initializer_for_final_bias(self):
        return tf.random_uniform_initializer(-3.0e-4, 3.0e-4)

    def create_base_network(self, inputs):
        # new setup
        weight_decay = 1e-2

        # dense layer1
        out = inputs
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
        out = tf.layers.dense(inputs=out, units=self.action_size,
                kernel_initializer=self.custom_initializer_for_final_dense(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                bias_initializer=self.custom_initializer_for_final_dense(),
                activation=None)

#        out = tf.nn.softsign(out)
        return out

    def train(self, training_step, q_gradient_batch, state_batch_list):

        feed_dict={self.q_gradient_input: q_gradient_batch}
        for idx, single_batch in enumerate(state_batch_list):
            feed_dict[self.frontend.map_inputs[idx]] = single_batch

        # Train the actor net
        if training_step%100 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = self.sess.run([self.summary_merged, self.update], feed_dict=feed_dict,
                                                                                options=run_options, run_metadata=run_metadata)
            self.summary_writer.add_run_metadata(run_metadata, 'a step%d' % training_step)
            self.summary_writer.add_summary(summary, training_step)
        else:
            self.sess.run(self.update, feed_dict=feed_dict)

    def get_action(self, state):
        feed_dict = {}
        for idx, single_state in enumerate(state):
            feed_dict[self.frontend.map_inputs[idx]] = [single_state]
        return self.sess.run(self.action_output, feed_dict=feed_dict)[0]

    def evaluate(self, state_batch_list):
        feed_dict = {}
        for idx, single_batch in enumerate(state_batch_list):
            feed_dict[self.frontend.map_inputs[idx]] = single_batch
        # Get an action batch
        actions = self.sess.run(self.action_output, feed_dict=feed_dict)
        return actions

    def target_evaluate(self, state_batch_list):
        feed_dict = {}
        for idx, single_batch in enumerate(state_batch_list):
            feed_dict[self.frontend.map_inputs[idx]] = single_batch
        # Get action batch
        actions = self.sess.run(self.action_output_target, feed_dict=feed_dict)
        return actions
