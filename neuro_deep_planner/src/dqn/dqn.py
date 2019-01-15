import numpy as np
import collections
import tensorflow as tf

# For saving replay buffer
import os
import sys
import time
import datetime

from value import QValueNetwork
#from dqn_data_manager import DQNDataManager
from dqn_replay_buffer import DQNReplayBuffer

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

# How often are we saving the net
SAVE_STEP = 10000

# Max training step
DEFAULT_MAX_TRAINING_STEP = 1e6

# Max training step with noise
MAX_NOISE_STEP = 3000000

# Epsilon for exploration
EPSILON_DECAY_LAST_FRAME =3e5
EPSILON_START = 1.0
EPSILON_FINAL = 0.1

class DQN:

    def __init__(self, session, state_shapes, layers, batch_size, action_bounds, action_res, data_path, data_manager=None, max_training_step=DEFAULT_MAX_TRAINING_STEP):

        self.layers = layers
        self.batch_size = batch_size
        self.data_path = data_path
        self.max_training_step = max_training_step

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
        self.action_dim = len(action_bounds)

        if isinstance(action_res, list):
            self.action_res = action_res
        else:
            self.action_res = [action_res]*self.action_dim

        self.action_space = np.prod(self.action_res)

        print("action_res:{}".format(self.action_res))
        print("action_space:{}".format(self.action_space))

        
        #self.action_table = np.array(range(self.action_space)).reshape(action_res)
        #self.action_table = np.meshgrid([range(a) for a in self.action_res], indexing='ij')
        print("action space======================================================")
        print([np.linspace(-1, 1, e) for e in self.action_res])
        self.action_table = np.meshgrid(*[np.linspace(-1, 1, e) for e in self.action_res], indexing='ij')
        print(self.action_table)
        self.id2action_lookup = [e.reshape(-1) for e in self.action_table]
        print(self.id2action_lookup)
        print("==================================================================")

        # Initialize our session
        self.session = session
        self.graph = self.session.graph

        with self.graph.as_default():
            #tf.set_random_seed(1)


            # Hardcode input size and action size
            self.depth = 4

            # Initialize the current action and the old action and old state for setting experiences
            self.old_state = [np.zeros((d, h, w), dtype='int8') for w, h, d in state_shapes]
            self.old_action = np.ones(2, dtype='float')
            self.network_action = np.zeros(2, dtype='float')
            self.action = np.zeros(2, dtype='float')

            # Initialize summary writers to plot variables during training
            #self.summary_op = tf.merge_all_summaries()
            self.summary_writer = tf.summary.FileWriter(self.tflog_path)

            # Initialize q value networks
            self.map_inputs = []
            for idx, shape in enumerate(state_shapes):
                d, h, w = shape
                self.map_inputs.append(tf.placeholder("float", [None, d, h, w], name="map_input_{}".format(idx)))
            self.training_step_variable = tf.Variable(0, name='global_step', trainable=False)

            self.episode_count_variable = tf.Variable(0, name='episode_count', trainable=False)
            self.episode_count_update = tf.assign(self.episode_count_variable, self.episode_count_variable + 1)

            self.q_value_network = QValueNetwork(self.map_inputs, self.layers, self.action_space, self.action_res, self.session, self.summary_writer, self.training_step_variable)

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
            self.data_manager = data_manager

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

            tf.train.start_queue_runners(sess=self.session)
            time.sleep(1)

            # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
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

            self.frame_counter = 0

    def train(self):
        # Check if the buffer is big enough to start training
        if self.data_manager.enough_data():

            # get the next random batch from the data manger
            state_batch, \
                action_batch, \
                reward_batch, \
                next_state_batch, \
                is_episode_finished_batch = self.data_manager.get_next_batch(self.batch_size)

            state_batch = np.divide(state_batch, 100.0)
            next_state_batch = np.divide(next_state_batch, 100.0)

            # Calculate y for the td_error of the critic
            y_batch = []
            # for double q
            #qa_value_array = self.actor_network.evaluate(next_state_batch)
            #next_action_batch = np.max(qa_value_array, axis=1)
            qa_value_array = self.q_value_network.target_evaluate(next_state_batch)
            #print(qa_value_array[0])
            q_value_batch = qa_value_array.max(axis=1, keepdims=True)
            #print("q_value_batch:{}".format(q_value_batch[0]))

            for i in range(0, BATCH_SIZE):
                if is_episode_finished_batch[i]:
                    y_batch.append([reward_batch[i]])
                else:
                    y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])

            # Now that we have the y batch lets train the critic
            self.q_value_network.train(self.training_step, y_batch, state_batch)

            # Update time step
            #self.training_step += 1
            self.training_step = self.session.run(self.training_step_variable)

            # Save model if necessary
            if self.training_step > 0 and self.training_step % SAVE_STEP == 0:
                st = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("[{}][{}] SAVE ###################################".format(st, self.training_step))
                self.saver.save(self.session, self.net_save_path + "/weights", global_step=self.training_step_variable)

        self.data_manager.check_for_enqueue()

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

    def is_running(self):
        if self.training_step > self.max_training_step:
            return False
        return True

    def get_action_from_action_id(self, action_id):
        assert (action_id < self.action_space)
        action = np.array([e[action_id] for e in self.id2action_lookup])
        return action

    def get_action(self, state):

        # normalize the state
        state = [s.astype(float) for s in state]
        state = [np.divide(s, 100.0) for s in state]

        # Get the action
        self.action_id = self.q_value_network.get_action(state)
        self.action = self.get_action_from_action_id(self.action_id)

        print("normalized action A", self.action)
        self.action = self.denormalize_action(self.action, self.action_bounds)

        print("self.noise_flag", self.noise_flag)
        print("self.training_step", self.training_step)
        print("action A", self.action)

        # Are we using noise?
        if self.noise_flag:

            epsilon = max(EPSILON_FINAL, EPSILON_START - float(self.training_step)/EPSILON_DECAY_LAST_FRAME)
            print('epsilon={}'.format(epsilon))
            if np.random.random() < epsilon:
                self.action_id = np.random.randint(0, self.action_space) # random sampling
                self.action = self.get_action_from_action_id(self.action_id)
                self.action = self.denormalize_action(self.action, self.action_bounds)
                print("action B", self.action)
            else:
                # do nothing
                pass
        #self.print_q_value(state, self.action)

        return self.action

    def set_experience(self, state, reward, is_episode_finished):

        if self.data_manager is None:
            raise ValueError("No data_manager is set.")

        # Make sure we're saving a new old_state for the first experience of every episode
        if self.first_experience:
            self.first_experience = False
        else:
            self.data_manager.store_experience_to_file(self.old_state, self.old_action_id, reward, state,
                                                       is_episode_finished)
            self.return_    = GAMMA*self.return_ + reward
            self.reward_sum = self.reward_sum + reward
            print("(return, reward_sum): ({}, {})".format(self.return_, self.reward_sum))

            # Uncomment if collecting data for the auto_encoder
            # experience = (self.old_state, self.old_action_id, reward, state, is_episode_finished)
            # self.buffer.append(experience)

        if is_episode_finished:
            print("*********************************************************************************************************************")
            print("a episode ends.({})".format(is_episode_finished))
            print("*********************************************************************************************************************")
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

        # Safe old state and old action for next experience
        self.old_state = state
        self.old_action_id = self.action_id
        print("-----------------------------------------------------------------------------------> frame_counter: {}".format(self.frame_counter))
        self.frame_counter += 1

    def print_q_value(self, state, action):

        string = "-"
        q_value = self.q_value_network.evaluate([state])
        stroke_pos = int(30 * q_value[0][0] + 30)
        if stroke_pos < 0:
            stroke_pos = 0
        elif stroke_pos > 60:
            stroke_pos = 60
        print '[' + stroke_pos * string + '|' + (60-stroke_pos) * string + ']', "Q: ", q_value[0][0], \
            "\tt: ", self.training_step
