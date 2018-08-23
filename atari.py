# 김성훈님 ( https://github.com/hunkim/ReinforcementZeroToAll/blob/master/07_3_dqn_2015_cartpole.py )
# 김태훈님 ( https://github.com/devsisters/DQN-tensorflow )
# https://gist.github.com/jcwleo
# http://www.modulabs.co.kr/RL_library/3652
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
INPUT = env.obseravtion_space.shape
OUTPUT = env.observation_space
HEIGHT = 84
WEIGHT = 84
LEARNING_RATE = 0.00025
DISCOUNT = 0.99
EPSILON = 0.01
MOMENTUM = 0.95

def get_init_state(frames, s):
    #episode 초기화
    for i in range(4):
        frames[:,:,:i] = preprocessing(s)


def cliped_error(err):
    return tf.where(tf.abs(err) < 1.0, 0.5 * tf.square(err), tf.abs(err) - 0.5)


def preprocessing(frame):
    return np.uint8(resize(rgb2gray(frame), (84, 84), mode='reflect') * 255)

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder=[]
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dst_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dst_vars):
        op_holder.append(dst_vars.assign(src_var.value()))
    return op_holder

class DQN:
    def __init__(self, sess, name):

        self.sess = sess
        self.num_episode = 3000
        self.num_action = 4
        self.num_frame = 4
        self.discount_factor = 0.9
        self.eps_decay = 0.95
        self.eps_interval = 10
        self.size_memory = 40000
        self.size_batch = 32
        self.num_fc = 256
        self.name = name

        self.forward()

    def forward(self):

        with tf.variable_scope(self.name):
            self.Y = tf.placeholder(tf.float32, [self.size_batch])
            self.action = tf.placeholder(tf.float32, [self.size_batch])
            self.frames = tf.placeholder(tf.float32, [self.size_batch, 84, 84, self.num_frame])

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


        self.one_hot = tf.one_hot(self.action, self.num_action,1.0,0.0)
        self.q_values = tf.reduce_sum(tf.multiply(fc2, self.one_hot))

        error = cliped_error(self.Y - self.q_values)
        self.loss = tf.reduce_mean(error)

        optimizer = tf.train.RMSPropOptimizer(0.00025, momentum=self.eps_decay, epsilon=0.01)
        self.train = optimizer.minimize(self.loss)


    def get_q(self, mem):
        return self.sess.run(self.q_values, feed_dict={self.frames : np.reshape(np.float32(mem/255), [-1, 84, 84, 4])})

    def get_action(self, q, e):
        if e > np.random.rand(1):
            action = np.random.randint(self.num_action)
        else:
            action = np.argmax(q)
        return action



def main():
    with tf.Session() as sess:
        mainDQN = DQN(sess, 'main')
        targetDQN = DQN(sess, 'target')

        sess.run(tf.global_variables_initializer())

        # weights copy operations
        copy_ops = get_copy_var_ops(dest_scope_name='target', src_scope_name='main')

        sess.run(copy_ops)

        recent_rlist = deque(maxlen=100)
        e = 1.
        episode, epoch, frame = 0, 0, 0

        epoch_score, epoch_Q = deque(), deque()
        average_Q, average_reward = deque(), deque()

        epoch_on = False
        replay_memory = deque(maxlen=40000)

        while epoch <= 3000:
            episode += 1

            frames = np.zeors([84,84,4], dtype = np.uint8)
            rall, count = 0, 0
            d = False
            ter = False

            s = env.reset()
            get_init_state(frames, s)

            while not d:
                env.render()

                frame+=1
                count+=1

                if e > 0.1 and frame > 50000:
                    e-= (1.-0.1)/1000000

                Q = mainDQN.get_q(frames[:,:,:4])
                average_Q .append(np.max(Q))

                action = mainDQN.get_action(Q,e)

                s1, r, d, l = env.step(action)

                ter = d
                reward = np.clip(r, -1, 1)

                frames[:,:,4] = preprocessing(s1)

                replay_memory.append( (np.copy(frames[:,:,:]), action, reward, ter))
                frames[:,:,4] = frames[:,:,1:]

                rall += r
