import tensorflow as tf


#
# reference: https://arxiv.org/abs/1511.04143
#
class GradInverter:

    def __init__(self, action_bounds, session):
        self.graph = session.graph
        with self.graph.as_default():
            self.sess = session

            self.action_size = len(action_bounds)
            self.action_input = tf.placeholder(tf.float32, [None, self.action_size])

            action_bounds_max = [ bound[1] for bound in action_bounds]
            action_bounds_min = [ bound[0] for bound in action_bounds]
            self.p_max = tf.constant(action_bounds_max, dtype=tf.float32)
            self.p_min = tf.constant(action_bounds_min, dtype=tf.float32)

            self.p_range = tf.constant([x - y for x, y in zip(action_bounds_max, action_bounds_min)],
                                       dtype=tf.float32)

            self.p_diff_max = tf.div(-self.action_input + self.p_max, self.p_range)
            self.p_diff_min = tf.div(self.action_input - self.p_min, self.p_range)

            self.zeros_act_grad_filter = tf.zeros([self.action_size])
            self.act_grad = tf.placeholder(tf.float32, [None, self.action_size])

            self.grad_inverter = tf.where(tf.greater(self.act_grad, self.zeros_act_grad_filter),
                                           tf.multiply(self.act_grad, self.p_diff_max),
                                           tf.multiply(self.act_grad, self.p_diff_min))

    def invert(self, grad, action):
        return self.sess.run(self.grad_inverter, feed_dict={self.act_grad: grad, self.action_input: action})
