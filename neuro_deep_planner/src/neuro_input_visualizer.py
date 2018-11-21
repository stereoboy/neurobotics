#!/usr/bin/env python

import rospy
import numpy as np
import sys

import cv2

from nav_msgs.msg import OccupancyGrid
from neuro_local_planner_wrapper.msg import Transition

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Global variable (not ideal but works)


class VisualizerRunnable(object):
    def __init__(self):
        self.fig = plt.figure("Transition Visualization", figsize=(16, 4))
        plt.title('Center Title')

        self.vis_im = plt.imshow(np.zeros(((86+10)+10, 86*4+30),
                            dtype='uint8'), cmap=plt.cm.gray, vmin=0, vmax=100,
                            interpolation="nearest", animated=True)
        self.subscriber = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/transition", Transition, self.callback, queue_size=1)
        self.costmap_subscriber = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/customized_costmap", OccupancyGrid, self.costmap_callback, queue_size=1)

        self.costmap_list = []

        self.move_img = None
        self.output = None
        self.costmap = None
        self.costmaps = None
        #self.vis_im = plt.imshow(np.array((106, 374)), animated=True)

    def update_im(self, *args):
        return [ self.vis_im ]

    def callback(self, transition):
        self.subscriber.unregister()

        if transition.is_episode_finished:
            print("reward after episode: {}".format(transition.reward))

        #print("-->{}chX{}px{}p".format(transition.depth, transition.width, transition.height))

        data_1d = np.asarray([(100 - d) for d in transition.state_representation])

        data_3d = data_1d.reshape(transition.depth, transition.height, transition.width).swapaxes(1, 2)

        data_3d = np.rollaxis(data_3d, 0, 3)

        # Make this a state batch with just one state in the batch

        if transition.reward >= 10.0:
#            h_divider = np.full((data_3d.shape[0], 10), 100)
#            v_divider = np.full((10, data_3d.shape[1]*4+30), 100)
            h_divider = np.full((data_3d.shape[0], 10), 75)
            v_divider = np.full((10, data_3d.shape[1]*4+30), 75)
            pass
        elif transition.reward < 0.0:
            h_divider = np.full((data_3d.shape[0], 10), 0)
            v_divider = np.full((10, data_3d.shape[1]*4+30), 0)
        else:
            h_divider = np.full((data_3d.shape[0], 10), 75)
            v_divider = np.full((10, data_3d.shape[1]*4+30), 75)

        output = v_divider

        h_stack = np.hstack((data_3d[:, :, 0], h_divider,
                             data_3d[:, :, 1], h_divider,
                             data_3d[:, :, 2], h_divider,
                             data_3d[:, :, 3]))
        v_stack = np.vstack((h_stack, v_divider))

        output = np.vstack((output, v_stack))

        self.move_img = np.sum(data_3d.astype(np.float32), axis=2)
        self.move_img = cv2.convertScaleAbs(self.move_img, alpha = 255.0/400.0, beta=0.0).astype(np.uint8)

        self.output = cv2.convertScaleAbs(output, alpha=2.55, beta=0.0)
        cv2.putText(self.output, str(transition.reward),(10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,50), 1)

        self.vis_im.set_data(output)
        self.subscriber = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/transition", Transition, self.callback, queue_size=1)

    def costmap_callback(self, data):
        w, h = data.info.width, data.info.height

        self.costmap = np.asarray([(100 - d) for d in data.data]).reshape((h, w)).astype(np.uint8)
        self.costmap = np.transpose(self.costmap, (1, 0))
        self.costmap = cv2.convertScaleAbs(self.costmap, alpha=2.55, beta=0.0)

        self.costmap_list.append(self.costmap.copy())
        if len(self.costmap_list) > 16:
            self.costmap_list.pop(0)

            total = []
            for i in range(4):
                row = []
                for j in range(4):
                    row.append(self.costmap_list[4*i + j].copy())
                    v_divider = np.full((row[0].shape[0], 10), 75, dtype=row[0].dtype)
                    row.append(v_divider)
                total.append(np.hstack(row))
                h_divider = np.full((10, total[0].shape[1]), 75, dtype=total[0].dtype)
                total.append(h_divider)
            self.costmaps = np.vstack(total)

    def run(self):
        #ani = animation.FuncAnimation(self.fig, self.update_im, interval=50, blit=True)
        #plt.show()

        cv2.namedWindow('transition')
        cv2.moveWindow('transition', 100, 100)
        #cv2.namedWindow('costmap')
        #cv2.moveWindow('costmap', 100, 100)
        #cv2.namedWindow('costmapx16')
        #cv2.moveWindow('costmapx16', 100, 100)
        cv2.namedWindow('move_img')
        cv2.moveWindow('move_img', 100, 100)
        counter = 0
        try:
            while not rospy.is_shutdown():
                if self.output is not None:
                    disp = cv2.resize(self.output.astype(np.uint8), (self.output.shape[1]*5, self.output.shape[0]*5))
                    cv2.imshow('transition', disp)
                    #cv2.imwrite('transition' + str(counter) + '.png', disp)
                    counter += 1
                    self.output = None
                #if self.costmap is not None:
                #    cv2.imshow('costmap', cv2.resize(self.costmap, (self.costmap.shape[1]*5, self.costmap.shape[0]*5)))
                #if self.costmaps is not None:
                #    cv2.imshow('costmapx16', cv2.resize(self.costmaps, (self.costmaps.shape[1]*4, self.costmaps.shape[0]*4)))
                if self.move_img is not None:
                    cv2.imshow('move_img', cv2.resize(self.move_img, (self.move_img.shape[1]*4, self.move_img.shape[0]*4)))
                ch = cv2.waitKey(1)
                if ch == 27:
                    break
        except rospy.ROSInterruptException:
            print("Goodbye~")
            sys.exit()

def main():

    rospy.init_node("neuro_input_visualizer", anonymous=False)

    visualizer = VisualizerRunnable()
    visualizer.run()

if __name__ == '__main__':
    main()
