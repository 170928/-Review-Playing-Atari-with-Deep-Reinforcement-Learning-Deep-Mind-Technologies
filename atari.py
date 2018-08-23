# 김성훈님 ( https://github.com/hunkim/ReinforcementZeroToAll/blob/master/07_3_dqn_2015_cartpole.py )
# 김태훈님 ( https://github.com/devsisters/DQN-tensorflow )
# https://gist.github.com/jcwleo
# http://www.modulabs.co.kr/RL_library/3652
# https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
# 코드를 참조했습니다.
import tensorflow as tf
import gym 
import numpy as np
import random as ran
from collections import deque
from argparse import ArgumentParser
from tensorflow.contrib.layers import xavier_initializer
from  skimage.color import rgb2gray
from skimage.transform import resize

env = gym.make('BreakoutDeterministic-v4')

MINIBATCH_SIZE = 32
HISTORY_SIZE = 4
TRAIN_START =  50000
FINAL_EXPLORATION = 0.1
TARGET_UPDATE = 10000
MEMORY_SIZE = 400000
EXPLORATION = 1000000
START_EXPLORATION = 1.
INPUT = env.observation_space
OUTPUT = env.action_space
HEIGHT = 80
WEIGHT = 105
LEARNING_RATE = 0.00025
DISCOUNT = 0.99
EPSILON = 0.01
MOMENTUM = 0.95


# Frame을 Convoludtion 2D를 위해 84x84 형태의 사각형으로 만들지 않는 방법입니다.
# 105 x 80 형태의 이미지가 입력이 됩니다.
def to_grayscale(img):
    return np.mean((img/255), axis=2).astype(np.uint8)

def downsample(img):
    return img[::2, ::2]

def preprocess(img):
    return to_grayscale(downsample(img))

# reward has to be -1 , 0,  1
def transform_reward(reward):
    return np.sign(reward)


def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder=[]
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dst_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dst_vars):
        op_holder.append(dst_vars.assign(src_var.value()))
    return op_holder

class DQN:
    def __init__(self, sess, name):


        self.forward()

    def forward(self):

        with tf.variable_scope(self.name):

            self.Y = tf.placeholder(tf.float32, [self.size_batch])
            self.action = tf.placeholder(tf.float32, [self.size_batch] )
            self.frames = tf.placeholder(tf.float32, [self.size_batch, HEIGHT, WEIGHT, self.num_frame])

            self.f1 = tf.get_variable(name='conv_w1', shape=[8,8,4,16], initializer=xavier_initializer())
            self.f2 = tf.get_variable(name='conv_w2', shape=[4,4,16, 32], initializer=xavier_initializer())

            h1 = tf.nn.conv2d(self.frames, self.f1, strides=[1,4,4,1], padding='VALID')
            h1 = tf.nn.relu(h1)
            h2 = tf.nn.conv2d(h1, self.f2, strides=[1,2,2,1], padding='VALID')
            h2 = tf.nn.relu(h2)
            h2 = tf.contrib.layers.flatten(h2)

            w1 = tf.get_variable(name='fc_w1',shape=[h2.get_shape().as_list()[1] ,self.num_fc], initializer=xavier_initializer())
            fc1 = tf.matmul(h2, w1)
            fc1 = tf.nn.relu(fc1)

            w2 = tf.get_variable(name='fc_w2', shape=[fc1.get_shape().as_list()[1], self.num_action], initializer=xavier_initializer())
            fc2 = tf.matmul(fc1, w2)


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