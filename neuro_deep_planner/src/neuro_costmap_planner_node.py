#!/usr/bin/env python

import rospy
import numpy as np
import cv2
import tensorflow as tf
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

from front_end import CostmapROSFrontEnd

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

    def run(self):
        print("###################################################################")
        print("mode: {}".format(FLAGS.mode))
        print("robot_type: {}".format(self.robot_type))
        print("logdir: {}".format(FLAGS.dir))
        print("###################################################################")

        self.agent = DDPG(FLAGS.mode, [(4, 86, 86)], action_bounds_dict[self.robot_type], FLAGS.dir)

        data = {'controller_frequency': self.controller_frequency, 'transition_frame_interval': self.transition_frame_interval}
        with open(os.path.join(self.agent.data_path, 'configuration.txt'), 'w') as f:
            json.dump(data, f, ensure_ascii=False)

        self.agent.noise_flag = True if FLAGS.mode == 'train' else False

        real_start_time = datetime.datetime.now() # real time
        last_msg_time = datetime.datetime.now()
        last_print_time = 0

        rate = rospy.Rate(11)
        try:
            while not rospy.is_shutdown():
                # check time
                real_current_time = datetime.datetime.now()
                if FLAGS.mode == 'train' and (real_current_time - last_msg_time).seconds >= TIMEOUT:
                    rospy.logerr("It's been over 60 seconds since the last data came in.")
                    sys.exit()
                elapsed_time = (real_current_time - real_start_time)
                if elapsed_time.seconds - last_print_time >= PRINT_INTERVAL:
                    last_print_time = math.floor(elapsed_time.seconds/PRINT_INTERVAL)*PRINT_INTERVAL
                    rospy.logwarn("#################################################################### %s"%(elapsed_time))

                if self.front_end.new_msg:
                    last_msg_time = datetime.datetime.now()
                    #self.front_end.show_input()
                    state, reward, done = self.front_end.step()
                    if not done:
                        # Send back the action to execute
                        if isinstance(state, list):
                            self.front_end.publish_action(self.agent.get_action(state))
                        else:
                            self.front_end.publish_action(self.agent.get_action([state]))

                    # Safe the past state and action + the reward and new state into the replay buffer
                    if FLAGS.mode == 'train':
                        self.agent.set_experience([state], reward, done)
                    self.front_end.new_msg = False
                # Train the network!
                if FLAGS.mode == 'train':
                    self.agent.train()

                if not self.agent.is_running():
                    rospy.logwarn("####################################################################")
                    rospy.logwarn("####################################################################")
                    rospy.logwarn("######################## E N D #####################################")
                    rospy.logwarn("####################################################################")
                    rospy.logwarn("####################################################################")
                    break

                #ch = cv2.waitKey(1)
                #rate.sleep()
                #if ch == 27:
                #    break
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
