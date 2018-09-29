#!/usr/bin/env python


import rospy
import numpy as np
from neuro_local_planner_wrapper.msg import Transition
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, Vector3


class ROSHandler:

    def __init__(self):

        # Initially assumed Input size, since init is false these values will be updated with the first received msg
        self.__init = False
        self.depth = 4
        self.height = 86
        self.width = 86

        self.state = np.zeros((self.width, self.height, self.depth), dtype='int8')

        self.reward = 0.0
        self.is_episode_finished = False

        self.robot_type = rospy.get_param('/robot_type', 'holonomic')

        self.__sub_move_base = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/transition", Transition,
                                                self.input_callback)
        self.__sub_setting = rospy.Subscriber("/noise_flag", Bool, self.setting_callback)
        self.__pub = rospy.Publisher("neuro_deep_planner/action", Twist, queue_size=10)
        self.__pub2 = rospy.Publisher("/ue4/robot/ctrl/move", Twist, queue_size=10)

        self.__new_msg_flag = False
        self.__new_setting_flag = False
        self.noise_flag = True

        self.controller_frequency = rospy.get_param("/move_base/controller_frequency");
        self.transition_frame_interval = rospy.get_param("/move_base/NeuroLocalPlannerWrapper/transition_frame_interval");


    def input_callback(self, transition_msg):

        # If msg is received for the first time adjust parameters

        if not self.__init:
            self.depth = transition_msg.depth
            self.width = transition_msg.width
            self.height = transition_msg.height
            self.state = np.zeros((self.depth, self.width, self.height), dtype='int8')
            self.__init = True

        # Lets update the new reward
        self.reward = transition_msg.reward




        # Check if episode is done or not
        self.is_episode_finished = transition_msg.is_episode_finished

        # Lets update the new costmap its possible that we need to switch some axes here...
        #if not self.is_episode_finished:
        self.state = np.asarray(transition_msg.state_representation, dtype='int8').reshape(self.depth, self.height, self.width)
        # We have received a new msg
        self.__new_msg_flag = True

    def setting_callback(self, setting_msg):

        # If msg is received for the first time adjust parameters

        self.noise_flag = setting_msg.data

        # We have received a setting
        self.__new_setting_flag = True

    def publish_action(self, action):

        # Generate msg output
        if self.robot_type == 'holonomic':
            vel_cmd = Twist(Vector3(action[0], action[1], 0), Vector3(0, 0, 0))
            #vel_cmd2 = Twist(Vector3(0.02*action[0], 0.02*action[1], 0), Vector3(0, 0, 0))
        elif self.robot_type == 'nonholonomic':
            vel_cmd = Twist(Vector3(action[0], 0, 0), Vector3(0, 0, action[1]))
            #vel_cmd2 = Twist(Vector3(0.02*action[0], 0, 0), Vector3(0, 0, action[1]))
        else:
            rospy.logerr("Wrong robot_type parameter")
            sys.exit(-1)

        # Send the action back
        self.__pub.publish(vel_cmd)
        self.__pub2.publish(vel_cmd)

    def new_msg(self):

        # Return true if new msg arrived only once for every new msg
        output = False
        if self.__new_msg_flag:
            output = True
            self.__new_msg_flag = False

        return output

    def new_setting(self):

        # Return true if new msg arrived only once for every new msg
        output = False
        if self.__new_setting_flag:
            output = True
            self.__new_setting_flag = False

        return output
