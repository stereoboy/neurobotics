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


class ActorNetwork:

    def __init__(self, map_input, action_size, session, summary_writer, training_step_variable):
        def getter_ema(ema):
            def ema_getter(getter, name, *args, **kwargs):
                var = getter(name, *args, **kwargs)
                ema_var = ema.average(var)
                print(var, ema_var)
                return ema_var if ema_var else var
            return ema_getter

        self.graph = session.graph

        with self.graph.as_default():

            # Get session and summary writer from ddpg
            self.sess = session
            self.summary_writer = summary_writer

            # Get input dimensions from ddpg
            self.action_size = action_size

            # Create actor network
            self.map_input = map_input
            #tf.summary.scalar('map_input', tf.reduce_mean(self.map_input[0]))
            self.q_gradient_input = tf.placeholder("float", [None, action_size], name='q_gradient_input')

            with tf.variable_scope('actor/network') as scope:
                self.action_output = self.create_base_network()

                # Get all the variables in the actor network for exponential moving average, create ema op
                self.actor_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
                print(self.actor_variables)

            #with tf.name_scope('actor/regularization'):
            #    regularization_loss = tf.losses.get_regularization_loss(scope='actor/network')

            # Define the gradient operation that delivers the gradients with the action gradient from the critic
            with tf.name_scope('actor/a_gradients'):
                self.parameters_gradients = tf.gradients(self.action_output, self.actor_variables, -self.q_gradient_input) #+ tf.gradients(regularization_loss, self.actor_variables, name="regularization")
                actor_gradient_summary = tf.summary.scalar('actor_gradients', tf.reduce_mean(self.parameters_gradients[0]))
            # Define the optimizer
            with tf.name_scope('actor/a_param_opt'):
                self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,
                                                                                       self.actor_variables), global_step=training_step_variable)

            with tf.control_dependencies([self.optimizer]):
                with tf.name_scope('actor/moving_average'):
                    self.ema_obj = tf.train.ExponentialMovingAverage(decay=TARGET_DECAY)
                    self.compute_ema = self.ema_obj.apply(self.actor_variables)

            # Create target actor network
            with tf.variable_scope('actor'):
                with tf.name_scope('target_network'):
                    with tf.variable_scope('network', reuse=tf.AUTO_REUSE, custom_getter=getter_ema(self.ema_obj)):
                        self.action_output_target = self.create_base_network()

            # Variables for plotting
            self.actions_mean_plot = [0, 0]
            self.summary_merged = tf.summary.merge([actor_gradient_summary])

    def custom_initializer_for_conv(self):
        return tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_in', distribution='uniform', seed=None, dtype=tf.float32)

    def custom_initializer_for_dense(self):
        return tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_in', distribution='uniform', seed=None, dtype=tf.float32)

    def custom_initializer_for_final_dense(self):
        return tf.random_uniform_initializer(-FINAL_WEIGHT_INIT, FINAL_WEIGHT_INIT)

    def custom_initializer_for_final_bias(self):
        return tf.random_uniform_initializer(-3.0e-4, 3.0e-4)

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

    def train(self, training_step, q_gradient_batch, state_batch):
        # Train the actor net
        if training_step%100 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = self.sess.run([self.summary_merged, self.compute_ema], feed_dict={self.q_gradient_input: q_gradient_batch, self.map_input: state_batch},
                                                                                options=run_options, run_metadata=run_metadata)
            self.summary_writer.add_run_metadata(run_metadata, 'a step%d' % training_step)
            self.summary_writer.add_summary(summary, training_step)
        else:
            self.sess.run(self.compute_ema, feed_dict={self.q_gradient_input: q_gradient_batch, self.map_input: state_batch})

    def get_action(self, state):
        return self.sess.run(self.action_output, feed_dict={self.map_input: [state]})[0]

    def evaluate(self, state_batch):
        # Get an action batch
        actions = self.sess.run(self.action_output, feed_dict={self.map_input: state_batch})
        return actions

    def target_evaluate(self, state_batch):
        # Get action batch
        actions = self.sess.run(self.action_output_target, feed_dict={self.map_input: state_batch})
        return actions
