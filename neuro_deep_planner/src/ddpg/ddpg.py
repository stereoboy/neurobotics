import numpy as np
import collections
from ou_noise import OUNoise
from critic import CriticNetwork
from frontend import FrontEndNetwork
from actor import ActorNetwork
from grad_inverter import GradInverter
import tensorflow as tf
#from data_manager import DataManager
from data_manager.replay_buffer import DQNReplayBuffer

# For saving replay buffer
import os
import time
import datetime

# Visualization
from state_visualizer import CostmapVisualizer


# How big are our mini batches
BATCH_SIZE = 32

# How big is our discount factor for rewards
GAMMA = 0.99

# How does our noise behave (MU = Center value, THETA = How strong is noise pulled to MU, SIGMA = Variance of noise)
MU = 0.0
THETA = 0.15
SIGMA = 0.20

# Should we load a saved net
PRE_TRAINED_NETS = False

# If we use a pretrained net
NET_LOAD_PATH = os.path.join(os.path.dirname(__file__), os.pardir)+"/pre_trained_networks/pre_trained_networks"

# Data Directory

# Should we use an existing initial buffer with experiences
NEW_INITIAL_BUFFER = False

# Visualize an initial state batch for debugging
VISUALIZE_BUFFER = False

# How often are we saving the net
SAVE_STEP = 10000

# Max training step with noise
MAX_NOISE_STEP = 3000000


class DDPG:

    def __init__(self, mode, state_shapes, action_bounds, data_path):

        self.mode = mode

        self.data_path = data_path

        # path to tensorboard data
        self.tflog_path = data_path + '/tf_logs'

        # path to experience files
        self.experience_path = data_path + '/experiences'

        # path to trained net files
        self.net_save_path = data_path + '/weights'

        # Make sure all the directories exist
        if not tf.gfile.Exists(self.data_path):
            tf.gfile.MakeDirs(self.data_path)
        if not tf.gfile.Exists(self.tflog_path):
            tf.gfile.MakeDirs(self.tflog_path)
        if not tf.gfile.Exists(self.experience_path):
            tf.gfile.MakeDirs(self.experience_path)
        if not tf.gfile.Exists(self.net_save_path):
            tf.gfile.MakeDirs(self.net_save_path)

        self.action_bounds = action_bounds

        # Initialize our session
#        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.session = tf.InteractiveSession()
        self.graph = self.session.graph

        with self.graph.as_default():
            #tf.set_random_seed(1)

            # View the state batches
            self.visualize_input = VISUALIZE_BUFFER
            if self.visualize_input:
                self.viewer = CostmapVisualizer()

            # Hardcode input size and action size
            self.depth = 4
            self.action_dim = len(action_bounds)

            # Initialize the current action and the old action and old state for setting experiences
            self.old_state = [np.zeros((d, h, w), dtype='int8') for w, h, d in state_shapes]
            self.old_action = np.ones(2, dtype='float')
            self.network_action = np.zeros(2, dtype='float')
            self.noise_action = np.zeros(2, dtype='float')
            self.action = np.zeros(2, dtype='float')

            # Initialize the grad inverter object to keep the action bounds
            self.grad_inv = GradInverter(self.action_bounds[0], self.action_bounds[1], self.session)

            # Initialize summary writers to plot variables during training
            #self.summary_op = tf.merge_all_summaries()
            self.summary_writer = tf.summary.FileWriter(self.tflog_path)

            # Initialize actor and critic networks
            self.map_inputs = []
            for idx, shape in enumerate(state_shapes):
                d, h, w = shape
                self.map_inputs.append(tf.placeholder("float", [None, d, h, w], name="map_input_{}".format(idx)))
            self.training_step_variable = tf.Variable(0, name='global_step', trainable=False)
            self.episode_count_variable = tf.Variable(0, name='episode_count', trainable=False)
            self.episode_count_update = tf.assign(self.episode_count_variable, self.episode_count_variable + 1)
            self.frontend_network0 = FrontEndNetwork('frontend0', self.map_inputs, self.session, self.summary_writer, self.training_step_variable)
            self.actor_network = ActorNetwork(self.frontend_network0, self.action_dim, self.session, self.summary_writer, self.training_step_variable)
            self.critic_network = CriticNetwork(self.map_inputs, self.action_dim, self.session, self.summary_writer)

            with tf.variable_scope('mean_return'):
                self.mean_return = tf.placeholder(tf.float32, name="mean_return")
                mean_return_summary = tf.summary.scalar("mean_return_val", self.mean_return)
                mean_return_summary_per_step = tf.summary.scalar("mean_return_val_per_step", self.mean_return)
            with tf.variable_scope('mean_reward_sum'):
                self.mean_reward_sum = tf.placeholder(tf.float32, name="mean_reward_sum")
                mean_reward_sum_summary = tf.summary.scalar("mean_reward_sum_val", self.mean_reward_sum)
                mean_reward_sum_summary_per_step = tf.summary.scalar("mean_reward_sum_val_per_step", self.mean_reward_sum)

            self.summary_merged0 = tf.summary.merge([mean_return_summary, mean_reward_sum_summary])
            self.summary_merged1 = tf.summary.merge([mean_return_summary_per_step, mean_reward_sum_summary_per_step])

            # Initialize the saver to save the network params
            self.saver = tf.train.Saver()

            # initialize the experience data manger
            if self.mode == 'train':
                #self.data_manager = DQNDataManager(BATCH_SIZE, self.experience_path, self.session)
                self.data_manager = DQNReplayBuffer(BATCH_SIZE, self.experience_path, self.session, max_memory_size=1000000, start_size=2000)

            # After the graph has been filled add it to the summary writer
            self.summary_writer.add_graph(self.graph)

            # Uncomment if collecting a buffer for the autoencoder
            # self.buffer = deque()

            # Should we load the pre-trained params?
            # If so: Load the full pre-trained net
            # Else:  Initialize all variables the overwrite the conv layers with the pretrained filters
            if PRE_TRAINED_NETS:
                self.saver.restore(self.session, NET_LOAD_PATH)
            else:
                checkpoint = tf.train.latest_checkpoint(self.net_save_path)
                if checkpoint:
                    print("Restoring from checkpoint: %s" % checkpoint)
                    self.saver.restore(self.session, checkpoint)
                else:
                    print("Couldn't find checkpoint to restore from. Starting over.")
                    self.session.run(tf.global_variables_initializer())
                    self.actor_network.init()
                    self.critic_network.init()

            tf.train.start_queue_runners(sess=self.session)
            time.sleep(1)

            # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
            self.exploration_noise = OUNoise(self.action_dim, MU, THETA, SIGMA)
            self.noise_flag = True

            # Initialize time step
            self.training_step = self.session.run(self.training_step_variable)
            print("training_step: {}".format(self.training_step))

            # Flag: don't learn the first experience
            self.first_experience = True

            self.episode_count = 0
            self.return_       = 0
            self.reward_sum    = 0
            self.total_returns = collections.deque(maxlen=100)
            self.total_rewards = collections.deque(maxlen=100)

    def train(self):
        # Check if the buffer is big enough to start training
        if self.data_manager.enough_data():
            print("train()")
            self.data_manager.check_for_enqueue()

            # get the next random batch from the data manger
            state_batch_list, \
                action_batch, \
                reward_batch, \
                next_state_batch_list, \
                is_episode_finished_batch = self.data_manager.get_next_batch()

            state_batch_list = [np.divide(state_batch, 100.0) for state_batch in state_batch_list]
            next_state_batch_list = [np.divide(next_state_batch, 100.0) for next_state_batch in next_state_batch_list]

            # Calculate y for the td_error of the critic
            y_batch = []
            next_action_batch = self.actor_network.target_evaluate(next_state_batch_list)
            q_value_batch = self.critic_network.target_evaluate(next_state_batch_list, next_action_batch)

            for i in range(0, BATCH_SIZE):
                if is_episode_finished_batch[i]:
                    y_batch.append([reward_batch[i]])
                else:
                    y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])

            # Now that we have the y batch lets train the critic
            self.critic_network.train(self.training_step, y_batch, state_batch_list, action_batch)

            # Get the action batch so we can calculate the action gradient with it
            # Then get the action gradient batch and adapt the gradient with the gradient inverting method
            action_batch_for_gradients = self.actor_network.evaluate(state_batch_list)
            q_gradient_batch = self.critic_network.get_q_gradient(state_batch_list, action_batch_for_gradients)
            q_gradient_batch = self.grad_inv.invert(q_gradient_batch, action_batch_for_gradients)

            # Now we can train the actor
            self.actor_network.train(self.training_step, q_gradient_batch, state_batch_list)

            # Update time step
            #self.training_step += 1
            self.training_step = self.session.run(self.training_step_variable)

            # Save model if necessary
            if self.training_step > 0 and self.training_step % SAVE_STEP == 0:
                st = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("[{}][{}] SAVE ###################################".format(st, self.training_step))
                self.saver.save(self.session, self.net_save_path + "/weights", global_step=self.training_step_variable)

    def normalize_action(self, raw_action, action_bounds):
        action = np.zeros_like(raw_action)
        for i, x in enumerate(raw_action):
            mean = 0.5*(action_bounds[i][0] + action_bounds[i][1])
            half_range = 0.5*(-action_bounds[i][0] + action_bounds[i][1])
            action[i] = float(x - mean)/half_range
#        print("normalize_action")
#        print("raw_action", raw_action)
#        print("action", action)
        return action

    def denormalize_action(self, action, action_bounds):
        raw_action = np.zeros_like(action)
        for i, x in enumerate(action):
            mean = 0.5*(action_bounds[i][0] + action_bounds[i][1])
            half_range = 0.5*(-action_bounds[i][0] + action_bounds[i][1])
            raw_action[i] = x*half_range + mean
#        print("recover_action")
#        print("action", action)
#        print("raw_action", raw_action)
        return raw_action

    def clip_action(self, action, action_bounds):
        clipped_action = np.zeros_like(action)
        for i, x in enumerate(action):
            clipped_action[i] = np.clip(x, action_bounds[i][0], action_bounds[i][1])
        return clipped_action

    def get_action(self, state):

        # normalize the state
        state = [s.astype(float) for s in state]
        state = [np.divide(s, 100.0) for s in state]

        # Get the action
        self.action = self.actor_network.get_action(state)

#        print("normalized action A", self.action)
#        self.action = self.denormalize_action(self.action, self.action_bounds)

        print("self.noise_flag", self.noise_flag)
        print("self.training_step", self.training_step)
        print("action A", self.action)

        # Are we using noise?
        if self.noise_flag:
            # scale noise down to 0 at training step 3000000
            freq_mod = int(1 + 8*(1 - np.exp(-5e-6*self.training_step)))
            noise_factor =max(0.4, np.exp(-5e-6*self.training_step))
            print("freq_mod: {}, noise_factor: {}".format(freq_mod, noise_factor))
            if self.training_step < MAX_NOISE_STEP and ((self.training_step)%freq_mod==0):
            #if self.training_step < MAX_NOISE_STEP:
                #self.action += (MAX_NOISE_STEP - self.training_step) / MAX_NOISE_STEP * self.exploration_noise.noise()
                #self.action += np.exp(-1e-5*self.training_step) * self.exploration_noise.noise()
                noise = noise_factor * self.exploration_noise.noise()
                print("noise", noise)
                self.action += noise
                print("action B", self.action)
            # if action value lies outside of action bounds, rescale the action vector
#            if self.action[0] < A0_BOUNDS[0] or self.action[0] > A0_BOUNDS[1]:
#                self.action[0] *= np.fabs(A0_BOUNDS[0]/self.action[0])
#            if self.action[1] < A0_BOUNDS[0] or self.action[1] > A0_BOUNDS[1]:
#                self.action[1] *= np.fabs(A1_BOUNDS[0]/self.action[1])
            self.action = self.clip_action(self.action, self.action_bounds)
#            print("action C", self.action)
        # Life q value output for this action and state
#        self.print_q_value(state, self.action)

        return self.action

    def set_experience(self, state, reward, is_episode_finished):

        # Make sure we're saving a new old_state for the first experience of every episode
        if self.first_experience:
            self.first_experience = False
        else:
            self.data_manager.store_experience_to_file(self.old_state, self.old_action, reward, state,
                                                       is_episode_finished)
            self.return_    = GAMMA*self.return_ + reward
            self.reward_sum = self.reward_sum + reward
            print("(return, reward_sum): ({}, {})".format(self.return_, self.reward_sum))

            # Uncomment if collecting data for the auto_encoder
            # experience = (self.old_state, self.old_action, reward, state, is_episode_finished)
            # self.buffer.append(experience)

        if is_episode_finished:
            self.total_returns.append(self.return_)
            self.total_rewards.append(self.reward_sum)
            _, self.episode_count = self.session.run([self.episode_count_update, self.episode_count_variable])
            print("self.episode_count:{}".format(self.episode_count))
            self.return_    = 0
            self.reward_sum = 0
            if self.episode_count%10 == 0:
                print("total_rewards:{}".format(self.total_rewards))
                if len(self.total_rewards) > 0:
                    mean_return     = np.mean(self.total_returns)
                    mean_reward_sum = np.mean(self.total_rewards)
                else:
                    mean_return     = 0
                    mean_reward_sum = 0
                summary0  = self.session.run(self.summary_merged0, feed_dict={self.mean_return: mean_return, self.mean_reward_sum: mean_reward_sum})
                self.summary_writer.add_summary(summary0, self.episode_count)
                summary1  = self.session.run(self.summary_merged1, feed_dict={self.mean_return: mean_return, self.mean_reward_sum: mean_reward_sum})
                self.summary_writer.add_summary(summary1, self.training_step)

            self.first_experience = True
            self.exploration_noise.reset()

        # Safe old state and old action for next experience
        self.old_state = state
        self.old_action = self.action
        #self.old_action = self.normalize_action(self.action, self.action_bounds)

    def print_q_value(self, state, action):

        string = "-"
        q_value = self.critic_network.evaluate([state], [action])
        stroke_pos = int(30 * q_value[0][0] + 30)
        if stroke_pos < 0:
            stroke_pos = 0
        elif stroke_pos > 60:
            stroke_pos = 60
        print '[' + stroke_pos * string + '|' + (60-stroke_pos) * string + ']', "Q: ", q_value[0][0], \
            "\tt: ", self.training_step
