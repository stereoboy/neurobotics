import tensorflow as tf
import math
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
LEARNING_RATE = 0.0005

# How much do we regularize the weights of the net
REGULARIZATION_DECAY = 0.0

# How fast does the target net track
TARGET_DECAY = 0.9999

# In what range are we initializing the weights in the final layer
FINAL_WEIGHT_INIT = 0.003

# How often do we plot variables during training
PLOT_STEP = 10


class CriticNetwork:

    def __init__(self, image_size, action_size, image_no, session, summary_writer):

        self.graph = session.graph

        with self.graph.as_default():

            # Get session and summary writer from ddpg
            self.sess = session
            self.summary_writer = summary_writer

            # Get input dimensions from ddpg
            self.image_size = image_size
            self.action_size = action_size
            self.image_no = image_no

            # Create critic network
            self.map_input = tf.placeholder("float", [None, image_size, image_size, image_no], name='map_input')
            self.action_input = tf.placeholder("float", [None, action_size], name="action_input")
            self.Q_output = self.create_network()

            # Get all the variables in the critic network for exponential moving average, create ema op
            with tf.variable_scope("critic") as scope:
                self.critic_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
            self.ema_obj = tf.train.ExponentialMovingAverage(decay=TARGET_DECAY)
            with tf.name_scope('critic/moving_average'):
                self.compute_ema = self.ema_obj.apply(self.critic_variables)

            # Create target critic network
            self.Q_output_target = self.create_target_network()

            # L2 Regularization for all Variables
#            self.regularization = 0
#            for variable in self.critic_variables:
#                self.regularization += tf.nn.l2_loss(variable)
            with tf.name_scope('critic/regularization'):
                regularization_loss = tf.losses.get_regularization_loss(scope='critic')

            # Define the loss with regularization term
            self.y_input = tf.placeholder("float", [None, 1], name="y_input")
            with tf.name_scope('cal_loss'):
                self.td_error = tf.reduce_mean(tf.pow(self.Q_output - self.y_input, 2))
                self.loss = self.td_error #+ regularization_loss

            # Define the optimizer
            with tf.name_scope('q_param_opt'):
                self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)

            # Define the action gradients for the actor training step
            with tf.name_scope('q_gradients'):
                self.action_gradients = tf.gradients(self.Q_output, self.action_input)

            # Variables for plotting
            self.action_grads_mean_plot = [0, 0]
            self.td_error_plot = 0

            # Training step counter (gets incremented after each training step)
            self.train_counter = 0

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
                kernel_initializer=self.custom_initializer_for_conv(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                activation=tf.nn.relu)
        # conv layer2
        out = tf.layers.conv2d(inputs=out, filters=FILTER2, kernel_size=RECEPTIVE_FIELD2, strides=STRIDE2, padding='VALID',
                kernel_initializer=self.custom_initializer_for_conv(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                activation=tf.nn.relu)
        # conv layer3
        out = tf.layers.conv2d(inputs=out, filters=FILTER3, kernel_size=RECEPTIVE_FIELD3, strides=STRIDE3, padding='VALID',
                kernel_initializer=self.custom_initializer_for_conv(),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                activation=tf.nn.relu)

        # dense layer1
        size = 1
        for n in out.get_shape().as_list()[1:]:
            size *= n
        out = tf.reshape(out, [-1, size])
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

    def create_network(self):

        with tf.variable_scope('critic'):
            out = self.create_base_network()
            print("out", out)

        return out

    def create_target_network(self):
        def getter_ema(ema):
            def ema_getter(getter, name, *args, **kwargs):
                var = getter(name, *args, **kwargs)
                ema_var = ema.average(var)
                print(var, ema_var)
                return ema_var if ema_var else var
            return ema_getter

        with tf.variable_scope('critic', reuse=True, custom_getter=getter_ema(self.ema_obj)):
            with tf.name_scope('target_network'):
                out = self.create_base_network()

        return out

    def restore_pretrained_weights(self, filter_path):

        # First restore the critic filters
        saver = tf.train.Saver({"weights_conv1": self.critic_variables[0],
                                "biases_conv1":  self.critic_variables[1],
                                "weights_conv2": self.critic_variables[2],
                                "biases_conv2":  self.critic_variables[3],
                                "weights_conv3": self.critic_variables[4],
                                "biases_conv3":  self.critic_variables[5],
                                # "weights_conv4": self.critic_variables[6],
                                # "biases_conv4":  self.critic_variables[7]
                                })

        saver.restore(self.sess, filter_path)

        # Now restore the target net filters
        saver_target = tf.train.Saver({"weights_conv1": self.ema_obj.average(self.critic_variables[0]),
                                       "biases_conv1":  self.ema_obj.average(self.critic_variables[1]),
                                       "weights_conv2": self.ema_obj.average(self.critic_variables[2]),
                                       "biases_conv2":  self.ema_obj.average(self.critic_variables[3]),
                                       "weights_conv3": self.ema_obj.average(self.critic_variables[4]),
                                       "biases_conv3":  self.ema_obj.average(self.critic_variables[5]),
                                       # "weights_conv4": self.ema_obj.average(self.critic_variables[6]),
                                       # "biases_conv4":  self.ema_obj.average(self.critic_variables[7])
                                       })

        saver_target.restore(self.sess, filter_path)

    def train(self, y_batch, state_batch, action_batch):

        # Run optimizer and compute some summary values
        td_error_value, _ = self.sess.run([self.td_error, self.optimizer], feed_dict={self.y_input: y_batch,
                                                                                      self.map_input: state_batch,
                                                                                      self.action_input: action_batch})

        # Now update the target net
        self.update_target()

        # Increment the td error plot variable for td error average plotting
        self.td_error_plot += td_error_value

        # Only save data every 10 steps
        if self.train_counter % PLOT_STEP == 0:

            self.td_error_plot /= PLOT_STEP

            # Add td error to the summary writer
            summary = tf.Summary(value=[tf.Summary.Value(tag='td_error_mean',
                                                         simple_value=np.asscalar(self.td_error_plot))])
            self.summary_writer.add_summary(summary, self.train_counter)

            self.td_error_plot = 0.0

        self.train_counter += 1

    def update_target(self):

        self.sess.run(self.compute_ema)

    def get_action_gradient(self, state_batch, action_batch):

        # Get the action gradients for the actor optimization
        action_gradients = self.sess.run(self.action_gradients, feed_dict={self.map_input: state_batch,
                                                                           self.action_input: action_batch})[0]

        # Create summaries for the action gradients and add them to the summary writer
        action_grads_mean = np.mean(action_gradients, axis=0)
        self.action_grads_mean_plot += action_grads_mean

        # Only save data every 10 steps
        if self.train_counter % PLOT_STEP == 0:

            self.action_grads_mean_plot /= PLOT_STEP

            summary_actor_grads_0 = tf.Summary(value=[tf.Summary.Value(tag='action_grads_mean[0]',
                                                                       simple_value=np.asscalar(
                                                                           self.action_grads_mean_plot[0]))])
            summary_actor_grads_1 = tf.Summary(value=[tf.Summary.Value(tag='action_grads_mean[1]',
                                                                       simple_value=np.asscalar(
                                                                           self.action_grads_mean_plot[1]))])
            self.summary_writer.add_summary(summary_actor_grads_0, self.train_counter)
            self.summary_writer.add_summary(summary_actor_grads_1, self.train_counter)

            self.action_grads_mean_plot = [0, 0]

        return action_gradients

    def evaluate(self, state_batch, action_batch):

        return self.sess.run(self.Q_output, feed_dict={self.map_input: state_batch, self.action_input: action_batch})

    def target_evaluate(self, state_batch, action_batch):
        return self.sess.run(self.Q_output_target, feed_dict={self.map_input: state_batch,
                                                              self.action_input: action_batch})
