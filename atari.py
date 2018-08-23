# 김성훈님 ( https://github.com/hunkim/ReinforcementZeroToAll/blob/master/07_3_dqn_2015_cartpole.py )
# 김태훈님 ( https://github.com/devsisters/DQN-tensorflow )
# https://gist.github.com/jcwleo
# http://www.modulabs.co.kr/RL_library/3652
# https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
# 코드를 참조했습니다.
import tensorflow as tf
import gym
import os
import numpy as np
import random as ran
from collections import deque
from argparse import ArgumentParser
from tensorflow.contrib.layers import xavier_initializer
from  skimage.color import rgb2gray
from skimage.transform import resize

env = gym.make('BreakoutDeterministic-v4')
IMGSIZE = 84


# Frame을 Convoludtion 2D를 위해 84x84 형태의 사각형으로.
class StateProcessor():
    def __init__(self):
        with tf.variable_scope("state_processor"):
            #[210, 160, 3]이미지의 input
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)
    def process(self, sess, state):
        #[84, 84, 1] state representing grayscale values.
        return sess.run(self.output, { self.input_state: state })


def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder=[]
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dst_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dst_vars):
        op_holder.append(dst_vars.assign(src_var.value()))
    return op_holder

class DQN:

    def __init__(self, sess, name, summaries_dir=None):

        self.summary_writer = None
        self.name = name
        self.num_hidden = 256
        self.num_action = 4
        self.size_batch = 32

        self.forward()

        if summaries_dir:
            summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            self.summary_writer = tf.summary.FileWriter(summary_dir)



    def forward(self):

        with tf.variable_scope(self.name):

            # The target value
            self.Y = tf.placeholder(tf.float32, [self.size_batch] )
            # Index of selected action
            self.action = tf.placeholder(tf.float32, [self.size_batch] )
            # Frame input 84x84x4
            self.X = tf.placeholder(tf.uint8, [self.size_batch, IMGSIZE, IMGSIZE, self.num_frame] )
            self.frames = tf.to_float(self.X)/255.0

            self.f1 = tf.get_variable(name='conv_w1', shape=[8,8,4,16], initializer=xavier_initializer())
            self.f2 = tf.get_variable(name='conv_w2', shape=[4,4,16,32], initializer=xavier_initializer())
            self.w1 = tf.get_variable(name='fc_w1',shape=[11*8*32 ,self.num_hidden], initializer=xavier_initializer())
            self.w2 = tf.get_variable(name='fc_w2', shape=[self.num_hidden ,self.num_action], initializer=xavier_initializer())

            h1 = tf.nn.conv2d(self.frames, self.f1, strides=[1,4,4,1], padding='VALID')
            h1 = tf.nn.relu(h1)
            h2 = tf.nn.conv2d(h1, self.f2, strides=[1,2,2,1], padding='VALID')
            h2 = tf.nn.relu(h2)
            # input의 형태가 [batch_size, ....]일때
            # input을 [batch_size, -1] 로 변환시켜 준다.
            h2 = tf.contrib.layers.flatten(h2)
            fc1 = tf.matmul(h2, w1)
            fc1 = tf.nn.relu(fc1)
            # out 변수는 DQN을 거친 결과로 [batch_size, num_action]
            self.pred_qval= tf.matmul(fc1, w2)

            # Get the predictions for the chosen actions only
            mask = tf.one_hot(action, 4, on_value=1, off_value=0)
            mask = tf.to_float(mask)

            pred_selected_qval = tf.multiply(self.pred_qval, mask)
            self.pred_action = tf.reduce_sum(pred_selected_qval, reduction_indices=[1])

            # Get loss
            self.losses = tf.squared_difference(self.Y, self.pred_action)
            self.loss = tf.reduce_mean(self.losses)

            # Optimalizer
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.95,0.0,1e-6)
            self.train_op = self.optimizer.minimize(self.loss)

            # Tensorboard
            self.summaries = tf.summary.merge([
                tf.summary.scalar("Loss", self.loss)
                tf.summary.scalar("Average Q Var", tf.reduce_mean(self.pred_action))
            ])

    def pred(self, sess, state):
        return sess.run(self.pred_qval, feed_dict={self.X : state})

    

if __name__ == "__main__":

    frame = env.reset()
    # Render
    env.render()

    is_done = False
    while not is_done:
        # Perform a random action, returns the new frame, reward and whether the game is over
        frame, reward, is_done, _ = env.step(env.action_space.sample())
        # Render
        env.render()