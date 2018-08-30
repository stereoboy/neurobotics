import tensorflow as tf
import numpy as np
import itertools
import os
import sys
import glob

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from data_manager import DataManager

# Parameters:
NUM_EXPERIENCES = 200       # How many experiences stored per file
MIN_FILE_NUM = 10           # How many files at minimum do we need for training

MIN_FILES_IN_QUEUE = 10     # Files are added when this Number is reached
NEW_FILES_TO_ADD = 200      # How many files are added to the fifo file queue


class DQNDataManager(DataManager):

    def __init__(self, batch_size, p_experience_path, session):
        super(DQNDataManager, self).__init__(batch_size=batch_size, p_experience_path=p_experience_path, session=session)

    # builds the pipeline how random batches are generated from the experience files on the hard drive
    def build_next_batch_op(self):
        reader = tf.TFRecordReader()

        _, serialized_experience = reader.read(self.filename_queue)

        # decode feature example
        features = tf.parse_single_example(serialized_experience, features={
            'state': tf.FixedLenFeature([], tf.string),
            'action': tf.FixedLenFeature([], tf.int64),
            'reward': tf.FixedLenFeature([], tf.float32),
            'next_state': tf.FixedLenFeature([], tf.string),
            'is_episode_finished': tf.FixedLenFeature([], tf.int64)})

        state = tf.decode_raw(features['state'], tf.uint8)
        state.set_shape([86*86*4])
        action = features['action']
        reward = features['reward']
        next_state = tf.decode_raw(features['next_state'], tf.uint8)
        next_state.set_shape([86*86*4])
        is_episode_finished = features['is_episode_finished']

        # reshape gridmaps
        state = tf.reshape(state, [86, 86, 4])
        next_state = tf.reshape(next_state, [86, 86, 4])

        # batch shuffling is done in a seperate thread
        # TODO tune the capacity in the following function call
        state_batch, action_batch, reward_batch, next_state_batch, is_episode_finished_batch = tf.train.shuffle_batch(
            [state, action, reward, next_state, is_episode_finished], batch_size=self.batch_size, capacity=10000,
            min_after_dequeue=100)

        return state_batch, action_batch, reward_batch, next_state_batch, is_episode_finished_batch

    def store_experience_to_file(self, state, action, reward, next_state, is_episode_finished):
        # write experience to file (s_t, a_t, r_t, s_{t+1})

        state_raw = state.tostring()
        next_state_raw = next_state.tostring()

        # build example
        example = tf.train.Example(features=tf.train.Features(feature={
            'state': tf.train.Feature(bytes_list=tf.train.BytesList(value=[state_raw])),
            'action': tf.train.Feature(int64_list=tf.train.Int64List(value=[action])),
            'reward': tf.train.Feature(float_list=tf.train.FloatList(value=[reward])),
            'next_state': tf.train.Feature(bytes_list=tf.train.BytesList(value=[next_state_raw])),
            'is_episode_finished': tf.train.Feature(int64_list=tf.train.Int64List(value=[is_episode_finished]))
        }))

        #self.writer.write(example.SerializeToString())
        self.write_buffer.append(example.SerializeToString())

        self.experience_counter += 1

        # store NUM_EXPERIENCES experiences per file
        if self.experience_counter == NUM_EXPERIENCES:
            print("[DataManager] write experiences into file -------------------------------------------------------------")
            try:
                # create new file
                self.filename = self.experience_path + '/data_' + str(self.file_counter) + '.tfrecords'
                self.writer = tf.python_io.TFRecordWriter(self.filename)

                # write the file to hdd
                for buf in self.write_buffer:
                    self.writer.write(buf)
                self.writer.close()
            except:
                if os.path.exists(self.filename):
                    os.remove(self.filename)
            print("[DataManager]------------------------------------------------------------------------------------------")

            # update counters
            self.write_buffer = []
            self.experience_counter = 0
            self.file_counter += 1

