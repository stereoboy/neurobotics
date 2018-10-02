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
import os
import datetime
import math

class FrontEnd(object):
    def __init__(self):
        raise NotImplementedError

    def get_msg_lock(self):
        raise NotImplementedError

    def build_state(self):
        raise NotImplementedError

    def show_input(self):
        raise NotImplementedError

    def calculate_reward(self, state):
        raise NotImplementedError

    def decide_done(self, state, reward):
        raise NotImplementedError

    def step(self):
        with self.get_msg_lock():
            print("step()")
            state   = self.build_state()
            reward  = self.calculate_reward(state)
            done    = self.decide_done(state, reward)
            return state, reward, done

    def publish_action(self, action):
        raise NotImplementedError

class CostmapROSFrontEnd(FrontEnd):
    def __init__(self):

        self.robot_type = rospy.get_param('/robot_type', 'holonomic')
        self._move_base = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/transition", Transition, self.transition_callback)
        self.thread_lock = threading.Lock()
        self.__pub = rospy.Publisher("neuro_deep_planner/action", Twist, queue_size=10)
        self.__pub2 = rospy.Publisher("/ue4/robot/ctrl/move", Twist, queue_size=10)

        # Initially assumed Input size, since init is false these values will be updated with the first received msg
        self.__init = False
        self.depth = 4
        self.height = 86
        self.width = 86

        self.new_msg = False

        self.msg_queue = collections.deque(maxlen=1)

    def get_msg_lock(self):
        return self.thread_lock


    def build_state(self):
        return self.state

    def calculate_reward(self, state):
        return self.reward

    def decide_done(self, state, reward):
        return self.is_episode_finished

    def show_input(self):
        disp = None
        with self.thread_lock:
            data_1d = np.asarray([(100 - d) for d in self.state])
            reward = self.reward 

        data_3d = data_1d.reshape(self.depth, self.height, self.width)

        if reward >= 10.0:
#            h_divider = np.full((data_3d.shape[0], 10), 100)
#            v_divider = np.full((10, data_3d.shape[1]*4+30), 100)
            h_divider = np.full((data_3d.shape[1], 10), 75)
            v_divider = np.full((10, data_3d.shape[2]*4+30), 75)
            pass
        elif reward < 0.0:
            h_divider = np.full((data_3d.shape[1], 10), 0)
            v_divider = np.full((10, data_3d.shape[2]*4+30), 0)
        else:
            h_divider = np.full((data_3d.shape[1], 10), 75)
            v_divider = np.full((10, data_3d.shape[2]*4+30), 75)

        output = v_divider

        h_stack = np.hstack((data_3d[0], h_divider,
                             data_3d[1], h_divider,
                             data_3d[2], h_divider,
                             data_3d[3]))
        v_stack = np.vstack((h_stack, v_divider))

        output = np.vstack((output, v_stack))

        move_img = np.sum(data_3d.astype(np.float32), axis=0)
        move_img = cv2.convertScaleAbs(move_img, alpha = 255.0/400.0, beta=0.0).astype(np.uint8)

        output = cv2.convertScaleAbs(output, alpha=2.55, beta=0.0)
        cv2.putText(output, str(reward),(10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,50), 1)

        cv2.imshow('costmap_transition',    cv2.resize(output, (output.shape[1]*4, output.shape[0]*4)))
        cv2.imshow('costmap_move',          cv2.resize(move_img, (move_img.shape[1]*4, move_img.shape[0]*4)))

    def transition_callback(self, transition_msg):
        # If msg is received for the first time adjust parameters
        if not self.__init:
            self.depth = transition_msg.depth
            self.width = transition_msg.width
            self.height = transition_msg.height
            self.state = np.zeros((self.depth, self.width, self.height), dtype='int8')
            self.__init = True

        with self.thread_lock:
            # Lets update the new reward
            self.reward = transition_msg.reward

            # Check if episode is done or not
            self.is_episode_finished = transition_msg.is_episode_finished

            # Lets update the new costmap its possible that we need to switch some axes here...
            #if not self.is_episode_finished:
            self.state = np.asarray(transition_msg.state_representation, dtype='int8').reshape(self.depth, self.height, self.width)
            # We have received a new msg
            self.new_msg = True

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

class VideoROSFrontEnd(FrontEnd):
    def __init__(self):

        self.bridge = CvBridge()
        self.cv_image = None
        self.thread_lock = threading.Lock()
        self.sub = rospy.Subscriber("/ue4/main_cam/label/image", Image, self.video_callback, queue_size=1)
        self.new_msg = False
        self.buffer = collections.deque(maxlen=FRAME_SIZE)

        self.h_divider  = None

    def get_msg_lock(self):
        return self.thread_lock

    def build_state(self):
        with self.thread_lock:
            # video state only
            if len(self.buffer) == FRAME_SIZE:
                transposed = [ np.transpose(x, (2, 0, 1)) for x in self.buffer]
                stacked = np.concatenate(transposed, axis=0)
                print(stacked.shape)
                return stacked

    def calculate_reward(self, state):
        return 0.0

    def decide_done(self, state, reward):
        return False

    def video_callback(self, img_msg):
        print("video_callback")
        self.cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        with self.thread_lock:
            self.buffer.append(self.cv_image)
        self.new_msg = True
    
    def show_input(self):
        print("show_input")
        disp = None
        with self.thread_lock:
            if len(self.buffer) == FRAME_SIZE:
                if self.h_divider is None:
                    self.h_divider = np.full((self.cv_image.shape[0], 10, self.cv_image.shape[2]), (0, 0, 0), dtype=np.uint8)

                temp = []
                move = np.zeros(self.buffer[0].shape, dtype=np.float32)
                for buf in self.buffer:
                    temp.append(buf)
                    temp.append(self.h_divider)
                    move += buf
                move /= FRAME_SIZE
                temp.append(move.astype(np.uint8))
                disp = np.hstack(temp)
        if disp is not None:
            print(disp.shape)
            print(disp.dtype)
            cv2.imshow('video_transition', disp)

class CombinedFrontEnd(FrontEnd):
    def __init__(self, *front_ends):
        self.front_ends = front_ends

    def get_msg_lock(self):
        raise NotImplementedError

    def step(self):
        with self.get_msg_lock():
            state   = self.build_state()
            reward  = self.calculate_reward(state)
            done    = self.decide_done(state, reward)
            return state, reward, done

    def publish_action(self, action):
        raise NotImplementedError

