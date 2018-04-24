import numpy as np
from ou_noise import OUNoise
from critic import CriticNetwork
from actor import ActorNetwork
import tensorflow as tf
from data_manager import DataManager

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

# Action boundaries
ACTION_BOUNDS = [[-0.4, 0.4], [-0.4, 0.4]]

# Should we load a saved net
PRE_TRAINED_NETS = False

# If we use a pretrained net
NET_LOAD_PATH = os.path.join(os.path.dirname(__file__), os.pardir)+"/pre_trained_networks/pre_trained_networks"

# Data Directory
#DATA_PATH = os.path.expanduser('~') + '/rl_nav_data'
DATA_PATH = './rl_nav_data'
#DATA_PATH = './rl_nav_data_simple_no_wall'

# path to tensorboard data
TFLOG_PATH = DATA_PATH + '/tf_logs'

# path to experience files
EXPERIENCE_PATH = DATA_PATH + '/experiences'

# path to trained net files
NET_SAVE_PATH = DATA_PATH + '/weights'

# Should we use an existing initial buffer with experiences
NEW_INITIAL_BUFFER = False

# Visualize an initial state batch for debugging
VISUALIZE_BUFFER = False

# How often are we saving the net
SAVE_STEP = 10000

# Max training step with noise
MAX_NOISE_STEP = 3000000


class DDPG:

    def __init__(self):

        # Make sure all the directories exist
        if not tf.gfile.Exists(TFLOG_PATH):
            tf.gfile.MakeDirs(TFLOG_PATH)
        if not tf.gfile.Exists(EXPERIENCE_PATH):
            tf.gfile.MakeDirs(EXPERIENCE_PATH)
        if not tf.gfile.Exists(NET_SAVE_PATH):
            tf.gfile.MakeDirs(NET_SAVE_PATH)

        # Initialize our session
        self.session = tf.Session()
        self.graph = self.session.graph

        with self.graph.as_default():
            tf.set_random_seed(1)

            # View the state batches
            self.visualize_input = VISUALIZE_BUFFER
            if self.visualize_input:
                self.viewer = CostmapVisualizer()

            # Hardcode input size and action size
            self.height = 86
            self.width = self.height
            self.depth = 4
            self.action_dim = 2

            # Initialize the current action and the old action and old state for setting experiences
            self.old_state = np.zeros((self.width, self.height, self.depth), dtype='int8')
            self.old_action = np.ones(2, dtype='float')
            self.network_action = np.zeros(2, dtype='float')
            self.noise_action = np.zeros(2, dtype='float')
            self.action = np.zeros(2, dtype='float')

            # Make sure the directory for the data files exists
            if not tf.gfile.Exists(DATA_PATH):
                tf.gfile.MakeDirs(DATA_PATH)

            # Initialize summary writers to plot variables during training
            #self.summary_op = tf.merge_all_summaries()
            self.summary_writer = tf.summary.FileWriter(TFLOG_PATH)

            # Initialize actor and critic networks
            self.training_step_variable = tf.Variable(0, name='global_step', trainable=False)
            self.actor_network = ActorNetwork(self.height, self.action_dim, self.depth, self.session,
                                              self.summary_writer, self.training_step_variable)
            self.critic_network = CriticNetwork(self.height, self.action_dim, self.depth, self.session,
                                                self.summary_writer)

            # Initialize the saver to save the network params
            self.saver = tf.train.Saver()

            # initialize the experience data manger
            self.data_manager = DataManager(BATCH_SIZE, EXPERIENCE_PATH, self.session)

            # Uncomment if collecting a buffer for the autoencoder
            # self.buffer = deque()

            # Should we load the pre-trained params?
            # If so: Load the full pre-trained net
            # Else:  Initialize all variables the overwrite the conv layers with the pretrained filters
            if PRE_TRAINED_NETS:
                self.saver.restore(self.session, NET_LOAD_PATH)
            else:
                checkpoint = tf.train.latest_checkpoint(NET_SAVE_PATH)
                if checkpoint:
                    print("Restoring from checkpoint: %s" % checkpoint)
                    self.saver.restore(self.session, checkpoint)
                else:
                    print("Couldn't find checkpoint to restore from. Starting over.")
                    self.session.run(tf.global_variables_initializer())

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

            # After the graph has been filled add it to the summary writer
            self.summary_writer.add_graph(self.graph)

    def train(self):
        # Check if the buffer is big enough to start training
        if self.data_manager.enough_data():

            # get the next random batch from the data manger
            state_batch, \
                action_batch, \
                reward_batch, \
                next_state_batch, \
                is_episode_finished_batch = self.data_manager.get_next_batch()

            state_batch = np.divide(state_batch, 100.0)
            next_state_batch = np.divide(next_state_batch, 100.0)

            # Are we visualizing the first state batch for debugging?
            # If so: We have to scale up the values for grey scale before plotting
            if self.visualize_input:
                state_batch_np = np.asarray(state_batch)
                state_batch_np = np.multiply(state_batch_np, -100.0)
                state_batch_np = np.add(state_batch_np, 100.0)
                self.viewer.set_data(state_batch_np)
                self.viewer.run()
                self.visualize_input = False

            # Calculate y for the td_error of the critic
            y_batch = []
            next_action_batch = self.actor_network.target_evaluate(next_state_batch)
            q_value_batch = self.critic_network.target_evaluate(next_state_batch, next_action_batch)

            for i in range(0, BATCH_SIZE):
                if is_episode_finished_batch[i]:
                    y_batch.append([reward_batch[i]])
                else:
                    y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])

            # Now that we have the y batch lets train the critic
            self.critic_network.train(y_batch, state_batch, action_batch)

            # Get the action batch so we can calculate the action gradient with it
            action_batch_for_gradients = self.actor_network.evaluate(state_batch)
            q_gradient_batch = self.critic_network.get_q_gradient(state_batch, action_batch_for_gradients)

            # Now we can train the actor
            self.actor_network.train(q_gradient_batch, state_batch)

            # Update time step
            #self.training_step += 1
            self.training_step = self.session.run(self.training_step_variable)

            # Save model if necessary
            if self.training_step > 0 and self.training_step % SAVE_STEP == 0:
                st = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("[{}][{}] SAVE ###################################".format(st, self.training_step))
                self.saver.save(self.session, NET_SAVE_PATH + "/weights", global_step=self.training_step_variable)

        self.data_manager.check_for_enqueue()

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

    def recover_action(self, action, action_bounds):
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
        state = state.astype(float)
        state = np.divide(state, 100.0)

        # Get the action
        self.action = self.actor_network.get_action(state)

        print("normalized action A", self.action)
        self.action = self.recover_action(self.action, ACTION_BOUNDS)

        print("self.noise_flag", self.noise_flag)
        print("self.training_step", self.training_step)
        print("action A", self.action)

        # Are we using noise?
        if self.noise_flag:
            # scale noise down to 0 at training step 3000000
            freq_mod = int(1 + 8*(1 - np.exp(-2e-5*self.training_step)))
            noise_factor = np.exp(-5e-5*self.training_step)
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
            self.action = self.clip_action(self.action, ACTION_BOUNDS)
            print("action C", self.action)
        # Life q value output for this action and state
        self.print_q_value(state, self.action)

        return self.action

    def set_experience(self, state, reward, is_episode_finished):

        # Make sure we're saving a new old_state for the first experience of every episode
        if self.first_experience:
            self.first_experience = False
        else:
            self.data_manager.store_experience_to_file(self.old_state, self.old_action, reward, state,
                                                       is_episode_finished)

            # Uncomment if collecting data for the auto_encoder
            # experience = (self.old_state, self.old_action, reward, state, is_episode_finished)
            # self.buffer.append(experience)

        if is_episode_finished:
            self.first_experience = True
            self.exploration_noise.reset()

        # Safe old state and old action for next experience
        self.old_state = state
        self.old_action = self.normalize_action(self.action, ACTION_BOUNDS)

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
