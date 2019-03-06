#!/usr/bin/env python

import rospy
import numpy as np
import cv2
import sys
import collections
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist, Vector3
from neuro_local_planner_wrapper.msg import Transition
import threading
import json
from ddpg.ddpg import DDPG
from dqn.dqn import DQN
# For saving replay buffer
import os
import datetime
import math
import argparse

from front_end import CostmapROSFrontEnd
from data_manager.replay_buffer import DQNReplayBuffer

TIMEOUT = 100
PRINT_INTERVAL = 10

# How big are our mini batches
BATCH_SIZE = 32

# Data Directory
DATA_PATH = os.path.expanduser('~') + '/rl_nav_data'

action_bounds_dict = {
    'holonomic': [[-0.4, 0.4], [-0.4, 0.4],],
    'nonholonomic':  [[-0.4, 0.4], [-0.1, 0.1]],
    }
#----------------------------------------------------------------------
FRAME_SIZE = 4

class PlannerNode(object):
    def __init__(self, front_end):
        self.front_end = front_end

        self.robot_type = rospy.get_param('/robot_type', 'holonomic')
        self.noise_flag = True


        self.controller_frequency = None
        self.transition_frame_interval = None
        if rospy.has_param("/move_base/controller_frequency"):
            self.controller_frequency = rospy.get_param("/move_base/controller_frequency")
        if rospy.has_param("/move_base/NeuroLocalPlannerWrapper/transition_frame_interval"):
            self.transition_frame_interval = rospy.get_param("/move_base/NeuroLocalPlannerWrapper/transition_frame_interval")

    def update_im(self, *args):
        return [ self.vis_im ]

    def init_tf_ddpg_agent(self):
        import tensorflow as tf

        actor_conv_layers = [(4, 2, 32), (4, 2, 32), (4, 2, 32)]
        #actor_conv_layers = [(4, 2, 64), (4, 2, 64), (4, 2, 64)]
        #actor_conv_layers = [(8, 4, 32), (4, 2, 64), (3, 1, 64)]
        #actor_conv_layers = [(8, 4, 32), (4, 2, 64), (4, 2, 64)]
        #actor_conv_layers = [(8, 4, 32), (4, 2, 64), (3, 1, 64)]
        actor_layers = (200, 200)

        critic_conv_layers = [(8, 4, 32), (4, 2, 64), (4, 3, 64)]
        #critic_conv_layers = [(8, 4, 64), (4, 2, 128), (4, 3, 128)]
        #critic_conv_layers = [(8, 4, 32), (4, 2, 64), (3, 1, 64)]
        critic_layers = (512, 512)

        layers = {'actor':{'frontend':actor_conv_layers, 'backend':actor_layers},
                'critic':{'frontend':critic_conv_layers, 'backend':critic_layers},}

        # Initialize tf session
#        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        session = tf.InteractiveSession()

        data_manager = DQNReplayBuffer(os.path.join(self.options.dir, 'experiences'), max_memory_size=1e6, start_size=2e3) if self.options.mode == 'train' else None
        action_bounds = action_bounds_dict[self.robot_type]
        agent = DDPG(session, [(4, 86, 86)], layers, BATCH_SIZE, action_bounds, self.options.dir, data_manager=data_manager, max_training_step=3e6)
        return agent

    def init_tf_dqn_agent(self):
        import tensorflow as tf

        value_conv_layers = [(8, 4, 32), (4, 2, 64), (3, 1, 64)]
        value_layers = (512, 512)

        layers = {'frontend':value_conv_layers, 'backend':value_layers}

        # Initialize tf session
#        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#        self.session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        session = tf.InteractiveSession()

        data_manager = DQNReplayBuffer(os.path.join(self.options.dir, 'experiences'), max_memory_size=1e6, start_size=5e4) if self.options.mode == 'train' else None
        action_bounds = action_bounds_dict[self.robot_type]
        agent = DQN(session, [(4, 86, 86)], layers, BATCH_SIZE, action_bounds, action_res=3, data_path=self.options.dir, data_manager=data_manager, max_training_step=3e6)
        return agent

    def run(self):

        parser = argparse.ArgumentParser(description='self controller based on costmap')
        parser.add_argument('--gui', action="store_true", default=False)
        parser.add_argument('--mode', action="store", dest='mode', default='train')
        parser.add_argument('--dir', action="store", dest='dir', default='./rl_nav_data')
        parser.add_argument('--agent', action="store", dest='agent', default='ddpg')

        self.options=parser.parse_args(sys.argv[1:])

        print("###################################################################")
        print("mode: {}".format(self.options.mode))
        print("robot_type: {}".format(self.robot_type))
        print("logdir: {}".format(self.options.dir))
        print("agent: {}".format(self.options.agent))
        print("###################################################################")

        if self.options.agent == 'ddpg':
            self.agent = self.init_tf_ddpg_agent()
        elif self.options.agent == 'dqn':
            self.agent = self.init_tf_dqn_agent()

        data = {'controller_frequency': self.controller_frequency, 'transition_frame_interval': self.transition_frame_interval}
        with open(os.path.join(self.agent.data_path, 'configuration.txt'), 'w') as f:
            json.dump(data, f, ensure_ascii=False)

        self.agent.noise_flag = True if self.options.mode == 'train' else False

        real_start_time = datetime.datetime.now() # real time
        last_msg_time = datetime.datetime.now()
        last_print_time = 0

        rate = rospy.Rate(11)
        try:
            while not rospy.is_shutdown():
                # check time
                real_current_time = datetime.datetime.now()
                if self.options.mode == 'train' and (real_current_time - last_msg_time).seconds >= TIMEOUT:
                    rospy.logerr("It's been over 60 seconds since the last data came in.")
                    sys.exit()
                elapsed_time = (real_current_time - real_start_time)
                if elapsed_time.seconds - last_print_time >= PRINT_INTERVAL:
                    last_print_time = math.floor(elapsed_time.seconds/PRINT_INTERVAL)*PRINT_INTERVAL
                    rospy.logwarn("#################################################################### %s"%(elapsed_time))

                if self.front_end.is_ready():
                    self.front_end.frame_begin()
                    last_msg_time = datetime.datetime.now()
                    if self.options.gui:
                        self.front_end.show_input()

                    state, reward, done = self.front_end.step()

                    if not done:
                        # Send back the action to execute
                        if isinstance(state, list):
                            self.front_end.publish_action(self.agent.get_action(state))
                        else:
                            self.front_end.publish_action(self.agent.get_action([state]))

                    # Safe the past state and action + the reward and new state into the replay buffer
                    if self.options.mode == 'train':
                        self.agent.set_experience([state], reward, done)
                    self.front_end.new_msg = False

                    self.front_end.frame_end()
                # Train the network!
                if self.options.mode == 'train':
                    self.agent.train()

                if not self.agent.is_running():
                    rospy.logwarn("####################################################################")
                    rospy.logwarn("####################################################################")
                    rospy.logwarn("######################## E N D #####################################")
                    rospy.logwarn("####################################################################")
                    rospy.logwarn("####################################################################")
                    break

                if self.options.gui:
                    ch = cv2.waitKey(1)
                    if ch == 27:
                        break
                #rate.sleep()

        except rospy.ROSInterruptException:
            sys.exit()

        rospy.logdebug("====================================")

def main():

    # Initialize the ANNs
    name="neuro_costmap_only_planner"
    rospy.init_node(name, anonymous=False, log_level=rospy.DEBUG)
    front_end = CostmapROSFrontEnd()
    node = PlannerNode(front_end=front_end)
    node.run()

if __name__ == '__main__':
    main()
