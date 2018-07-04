#!/usr/bin/env python

import rospy
from ros_handler import ROSHandler
from ddpg import DDPG
# For saving replay buffer
import os
import tensorflow as tf
import datetime
import math

TIMEOUT = 60
PRINT_INTERVAL = 10

# Do we want to plot results during training
PLOTTING = False         # Do we want to plot
PLOT_TIME = 1800        # How long do we want to plot in seconds, e.g. 60 s/min * 30 min = 1800 s
PLOT_INTERVALL = 50000  # How many training steps until we plot the results

# Data Directory
DATA_PATH = os.path.expanduser('~') + '/rl_nav_data'

# path to tensorboard data
PLOT_PATH = DATA_PATH + '/plotting_results'

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
    print("###################################################################")

    agent = DDPG(FLAGS.mode, action_bounds_dict[ros_handler.robot_type], FLAGS.dir)

    import json
    data = {'controller_frequency': ros_handler.controller_frequency, 'transition_frame_interval': ros_handler.transition_frame_interval}
    with open(os.path.join( agent.data_path, 'configuration.txt'), 'w') as f:
        json.dump(data, f, ensure_ascii=False)


    if FLAGS.mode == 'eval':
        agent.noise_flag = False
    else:
        agent.noise_flag = True

    # For plotting
    currently_plotting = False
    goal_count = 0
    crash_count = 0
    start_time = rospy.get_time() # simulation speedup time


    real_start_time = datetime.datetime.now() # real time
    last_msg_time = datetime.datetime.now()
    last_print_time = 0

    # Make sure the directory for the plotting exists
    if not tf.gfile.Exists(PLOT_PATH):
        tf.gfile.MakeDirs(PLOT_PATH)
    f = open(PLOT_PATH + '/results', 'w')

    while not rospy.is_shutdown():

        # If we are plotting results we don't want to train and need to turn of noise!
        if PLOTTING and not currently_plotting and agent.training_step > 0 and \
                                agent.training_step % PLOT_INTERVALL == 0:
            currently_plotting = True
            agent.noise_flag = False
            start_time = rospy.get_time()

        if currently_plotting and rospy.get_time() - start_time > PLOT_TIME:
            # Plot the results
            string = str(agent.training_step) + ', ' + str(goal_count) + ', ' + str(crash_count) + '\n'
            f.write(string)

            # Reset all parameters
            currently_plotting = False
            agent.noise_flag = True
            goal_count = 0
            crash_count = 0

        # If we are plotting results we need to count reached goals and crashes
        if currently_plotting:
            # Count the positive and negative rewards
            if ros_handler.new_msg():
                if not ros_handler.is_episode_finished:
                    # Send back the action to execute
                    ros_handler.publish_action(agent.get_action(ros_handler.state))
                elif ros_handler.reward == 1:
                    goal_count += 1
                elif ros_handler.reward == -1:
                    crash_count += 1

        # If we're not plotting results
        else:
            # If we have a new msg we might have to execute an action and need to put the new experience in the buffer
            real_current_time = datetime.datetime.now()
            if FLAGS.mode == 'train' and (real_current_time - last_msg_time).seconds >= TIMEOUT: 
                rospy.logerr("It's been over 60 seconds since the last data came in.")
                import sys
                sys.exit()

            elapsed_time = (real_current_time - real_start_time)
            if elapsed_time.seconds - last_print_time >= PRINT_INTERVAL:
                last_print_time = math.floor(elapsed_time.seconds/PRINT_INTERVAL)*PRINT_INTERVAL
                rospy.logwarn("#################################################################### %s"%(elapsed_time))

            if ros_handler.new_msg():
                last_msg_time = datetime.datetime.now()
                if not ros_handler.is_episode_finished:
                    # Send back the action to execute
                    ros_handler.publish_action(agent.get_action(ros_handler.state))

                # Safe the past state and action + the reward and new state into the replay buffer
                if FLAGS.mode == 'train':
                    agent.set_experience(ros_handler.state, ros_handler.reward, ros_handler.is_episode_finished)

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
