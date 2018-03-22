#!/usr/bin/env python

import rospy
import numpy as np

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

        #self.vis_im = plt.imshow(np.array((106, 374)), animated=True)

    def update_im(self, *args):
        return [ self.vis_im ]

    def callback(self, data):
        self.subscriber.unregister()

        if not data.is_episode_finished:

            #print("-->{}chX{}px{}p".format(data.depth, data.width, data.height))

            data_1d = np.asarray([(100 - d) for d in data.state_representation])

            data_3d = data_1d.reshape(data.depth, data.height, data.width).swapaxes(1, 2)

            data_3d = np.rollaxis(data_3d, 0, 3)

            # Make this a state batch with just one state in the batch
            data_3d = np.expand_dims(data_3d, axis=0)

            h_divider = np.full((data_3d.shape[1], 10), 75)
            v_divider = np.full((10, data_3d.shape[1]*4+30), 75)

            output = v_divider

            for data in data_3d:

                h_stack = np.hstack((data[:, :, 0], h_divider,
                                     data[:, :, 1], h_divider,
                                     data[:, :, 2], h_divider,
                                     data[:, :, 3]))

                v_stack = np.vstack((h_stack, v_divider))

                output = np.vstack((output, v_stack))

            self.vis_im.set_data(output)
        self.subscriber = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/transition", Transition, self.callback, queue_size=1)

    def run(self):
        ani = animation.FuncAnimation(self.fig, self.update_im, interval=50, blit=True)
        plt.show()

def main():

    rospy.init_node("neuro_input_visualizer", anonymous=False)

    visualizer = VisualizerRunnable()
    visualizer.run()

if __name__ == '__main__':
    main()
