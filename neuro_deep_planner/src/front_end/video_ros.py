import rospy
import numpy as np
import cv2
import tensorflow as tf
import sys
import collections
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import ChannelFloat32
from geometry_msgs.msg import Twist, Vector3, PolygonStamped
from neuro_local_planner_wrapper.msg import Transition
import threading
import json
import os
import glob
import datetime
import math
from front_end import FrontEnd

FRAME_SIZE = 4

Target = collections.namedtuple('Target', field_names=['color', 'loc'])
class VideoROSFrontEnd(FrontEnd):
    def __init__(self, target_img_dir):

        self.robot_type = rospy.get_param('/robot_type', 'holonomic')
        self.bridge = CvBridge()
        self.cv_image = None
        self.thread_lock = threading.Lock()
        self.buffer = collections.deque(maxlen=FRAME_SIZE)
        self.sub_video = rospy.Subscriber("/ue4/main_cam/label/image", Image, self.video_callback, queue_size=1)
        self.new_msg = False

        self.h_divider  = None

        self.__pub = rospy.Publisher("neuro_deep_planner/action", Twist, queue_size=10)
        self.__pub2 = rospy.Publisher("/ue4/robot/ctrl/move", Twist, queue_size=10)

        self.shapes = [(FRAME_SIZE*3, 240/2, 320/2)]
        h, w = self.shapes[0][1:]

        self.target_marker_img = cv2.imread(os.path.join(target_img_dir, 'target_marker_img.png'))
        self.target_marker_img = cv2.resize(self.target_marker_img, (w, h), interpolation=cv2.INTER_NEAREST)
        self.target_img_infos = [   
                                    {"filename":'target_img.png', 'coeffs':[10.0, 0.0, 0.0], 'png':None},
#                                    {"filename":'target_img_bg.png', 'coeffs':[0.1, 0.0, -0.1], 'png':None},
                                ]
        print("load target images.... @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
#        for filepath in glob.glob(os.path.join(target_img_dir, "target_img.png")):
#            target_img = cv2.imread(filepath)
#            print("\ttarget image: {} ({})".format(filepath, target_img.shape))
#            self.target_imgs.append(target_img)
        for info in self.target_img_infos:
            filepath = os.path.join(target_img_dir, info['filename'])
            target_img = cv2.imread(filepath)
            print("\ttarget image: {} ({})".format(filepath, target_img.shape))
            target_img = cv2.resize(target_img, (w, h), interpolation=cv2.INTER_NEAREST)
            info['png'] = target_img
            info['coeffs'][1] = 1.0/np.sum((target_img != (0, 0, 0)).all(axis=2))
        print(self.target_img_infos)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        self.targets   = [Target((191, 105, 112),((30, 20), (70, 100))), Target((72, 121, 89), ((90, 20), (130, 100)))]

        self.frame_id = 0
        self.status_cnt = 0
        self.debug_frame = 0

    def get_msg_lock(self):
        return self.thread_lock

    def build_state(self):
        # video state only
        if len(self.buffer) == FRAME_SIZE:
            transposed = [ np.transpose(x, (2, 0, 1)) for x in self.buffer]
            stacked = np.concatenate(transposed, axis=0)
            return stacked

    def calculate_reward(self, state):
#        for target in self.targets:
#            roi = self.cv_image[target.loc[0][1]:target.loc[1][1], target.loc[0][0]:target.loc[1][0]]
#        roi = self.cv_image[20:100, 30:70]
#        countA = np.sum(roi==(191,105,112))
#        print(countA)
#        countB = np.sum(self.cv_image==(191,105,112)) - countA
#        print(countB)
#
#        print(countA - countB)
#        self.reward_disp = cv2.rectangle(self.cv_image.copy(),  self.targets[0].loc[0], self.targets[0].loc[1], (255,255,255), 1)
#        self.reward_disp = cv2.rectangle(self.reward_disp,      self.targets[1].loc[0], self.targets[1].loc[1], (255,255,255), 1)
#        cv2.putText(self.reward_disp, str(countA),(10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,50), 1)
#        cv2.putText(self.reward_disp, str(countB),(10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,50), 1)
#        cv2.putText(self.reward_disp, str(countA - countB),(10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,50), 1)
#        cv2.imshow('reward', disp)
        total_reward = 0
        self.reward_disps = []
        for info in self.target_img_infos:
            target_img = info['png']
            coeffs = info['coeffs']
            diff = cv2.absdiff(self.cv_image, target_img)
            mask = np.sum(diff, axis=2)
            mask = (mask == 0.0)
            mask = mask.astype(np.uint8)
            reward = coeffs[0]*coeffs[1]*np.sum(mask) + coeffs[2]
            img = cv2.bitwise_and(self.cv_image, self.cv_image, mask=mask)
            cv2.putText(img, str(reward),(10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
            cv2.putText(img, str(self.done),(10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
            self.reward_disps.append(target_img)
            self.reward_disps.append(img)
            total_reward += reward
        print("reward: {}".format(reward))
        return total_reward

    def decide_done(self, state, reward):
        return self.done

    def video_callback(self, img_msg):
        print("[%d]>>>video_callback(%d)"%(threading.current_thread().ident, img_msg.header.seq))
        with self.thread_lock:
            self.cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            print(self.cv_image.shape)
            h, w = self.shapes[0][1:]
            print(h, w)
            self.cv_image = cv2.resize(self.cv_image, (w, h), interpolation=cv2.INTER_NEAREST)
            self.cv_image = cv2.add(self.cv_image, self.target_marker_img)
            self.buffer.append(self.cv_image)

            # FRAME END #
            if len(self.buffer) == FRAME_SIZE:
                self.new_msg = True
            self.frame_id += 1
            print("frame_id: %d"%(self.frame_id))
            print("<<<video_callback()")

    def status_callback(self, msg):
        print(threading.current_thread())
        print("[%d]>>>status_callback(%d)"%(threading.current_thread().ident, msg.header.seq))
        with self.thread_lock:
            seq = int(msg.polygon.points[0].x)
            self.done   = (msg.polygon.points[0].y!=0.0)
            self.reward = msg.polygon.points[0].z

            if self.done:
                self.debug_frame = self.frame_id

            if (self.frame_id - self.debug_frame) < 5:
                temp_img = self.cv_image.copy()
                cv2.putText(temp_img, str(self.done),(10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
                cv2.putText(temp_img, str(self.reward),(10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
#                cv2.imwrite(("./temp/%05d.png"%(self.frame_id)), temp_img)
            self.status_cnt += 1
            print("status_cnt:%d"%(self.status_cnt))
            print("<<<status_callback()")

    def show_input(self):
        print(">>>show_input")
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
                for img in self.reward_disps:
                    temp.append(self.h_divider)
                    temp.append(img)
                disp = np.hstack(temp)
        if disp is not None:
            h, w = disp.shape[:2]
            disp = cv2.resize(disp, (w*2, h*2), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('video_transition', disp)

#            if (self.frame_id - self.debug_frame) < 5:
#                cv2.imwrite(("./temp2/%05d.png"%(self.frame_id)), disp)

        print("<<<show_input")

    def is_ready(self):
        return self.new_msg

    def frame_begin(self):
        pass

    def frame_end(self):
        self.new_msg = False

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

        vel_cmd = Twist(Vector3(action[0]*10.0, 0, 0), Vector3(0, 0, action[1]))
        # Send the action back
        #self.__pub.publish(vel_cmd)
        self.__pub2.publish(vel_cmd)

    def reset(self):
        self.buffer.clear()

