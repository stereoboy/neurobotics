#!/usr/bin/env python

import rospy
from ros_handler import ROSHandler
from ddpg import DDPG
from dqn.dqn import DQN
# For saving replay buffer
import os
import tensorflow as tf
import datetime
import math
import json
import sys

TIMEOUT = 100
PRINT_INTERVAL = 10

# Data Directory
DATA_PATH = os.path.expanduser('~') + '/rl_nav_data'

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("mode", "train", "mode: train or eval")
tf.flags.DEFINE_string("dir", "./rl_nav_data", "directory to save")

action_bounds_dict = {
    'holonomic': [[-0.4, 0.4], [-0.4, 0.4]],
    'nonholonomic':  [[-0.4, 0.4], [-0.1, 0.1]],
    }

def main():

    # Initialize the ANNs
    rospy.init_node("neuro_deep_planner", anonymous=False)

    ros_handler = ROSHandler()
    ros_handler.on_policy = False

    print("###################################################################")
    print("mode: {}".format(FLAGS.mode))
    print("robot_type: {}".format(ros_handler.robot_type))
    print("###################################################################")

    agent = DDPG(FLAGS.mode, [(4, 86, 86)], action_bounds_dict[ros_handler.robot_type], FLAGS.dir)
    #agent = DQN(FLAGS.mode, action_bounds_dict[ros_handler.robot_type], 3, FLAGS.dir)

    data = {'controller_frequency': ros_handler.controller_frequency, 'transition_frame_interval': ros_handler.transition_frame_interval}
    with open(os.path.join( agent.data_path, 'configuration.txt'), 'w') as f:
        json.dump(data, f, ensure_ascii=False)


    if FLAGS.mode == 'eval':
        agent.noise_flag = False
    else:
        agent.noise_flag = True

    real_start_time = datetime.datetime.now() # real time
    last_msg_time = datetime.datetime.now()
    last_print_time = 0

    while not rospy.is_shutdown():

        # If we have a new msg we might have to execute an action and need to put the new experience in the buffer
        real_current_time = datetime.datetime.now()
        if FLAGS.mode == 'train' and (real_current_time - last_msg_time).seconds >= TIMEOUT:
            rospy.logerr("It's been over 60 seconds since the last data came in.")
            sys.exit()

        elapsed_time = (real_current_time - real_start_time)
        if elapsed_time.seconds - last_print_time >= PRINT_INTERVAL:
            last_print_time = math.floor(elapsed_time.seconds/PRINT_INTERVAL)*PRINT_INTERVAL
            rospy.logwarn("#################################################################### %s"%(elapsed_time))

        if ros_handler.new_msg():
            last_msg_time = datetime.datetime.now()
            if not ros_handler.is_episode_finished:
                # Send back the action to execute
                ros_handler.publish_action(agent.get_action([ros_handler.state]))

            # Safe the past state and action + the reward and new state into the replay buffer
            if FLAGS.mode == 'train':
                agent.set_experience([ros_handler.state], ros_handler.reward, ros_handler.is_episode_finished)
#                    agent.train()
#                    agent.train()

        elif ros_handler.new_setting():

            agent.noise_flag = ros_handler.noise_flag
            print("noise_flag: {}".format(agent.noise_flag))

        else:
            # Train the network!
            if FLAGS.mode == 'train':
                agent.train()
            else:
                pass


if __name__ == '__main__':
    main()
