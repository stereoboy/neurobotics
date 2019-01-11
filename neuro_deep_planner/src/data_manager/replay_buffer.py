import tensorflow as tf
import numpy as np
import collections
import itertools
import os
import sys
import glob

# Parameters:
#MAX_MEMORY_SIZE  = 1e6
#START_SIZE       = 5e4
MAX_MEMORY_SIZE  = 1e6
START_SIZE       = 2000


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'is_episode_finished'])
class DQNReplayBuffer(object):

    def __init__(self, experience_path, max_memory_size=MAX_MEMORY_SIZE, start_size=START_SIZE):

        self.max_memory_size    = max_memory_size
        self.start_size         = start_size
        self.buffer = collections.deque(maxlen=max_memory_size)

    def enough_data(self):
        if len(self.buffer) >= self.start_size:
            return True
        else:
            return False

    # builds the pipeline how random batches are generated from the experience files on the hard drive
    def get_next_batch(self, batch_size):
        sampled_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in sampled_indices])

        state_batch_list          = [np.array(s,            dtype=np.int8) for s in zip(*states)]
        action_batch              = np.array(actions,       dtype=np.float32)
        reward_batch              = np.array(rewards,       dtype=np.float32)
        next_state_batch          = [np.array(s,            dtype=np.int8) for s in zip(*next_states)]
        is_episode_finished_batch = np.array(dones,         dtype=np.int64)

        return state_batch_list, action_batch, reward_batch, next_state_batch, is_episode_finished_batch

    def store_experience_to_file(self, state, action, reward, next_state, is_episode_finished):
        #print(state, action, reward, next_state, is_episode_finished)
        self.buffer.append(Experience(state, action, reward, next_state, is_episode_finished))
        if len(self.buffer)%100 == 0:
            print("------------------------------------------------------------------------------------>current_buffer_size: {}".format(len(self.buffer)))
    
    def check_for_enqueue(self):
        pass
