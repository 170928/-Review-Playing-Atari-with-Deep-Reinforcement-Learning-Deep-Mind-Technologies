# 김성훈님 ( https://github.com/hunkim/ReinforcementZeroToAll/blob/master/07_3_dqn_2015_cartpole.py )
# 김태훈님 ( https://github.com/devsisters/DQN-tensorflow )
# https://gist.github.com/jcwleo
# http://www.modulabs.co.kr/RL_library/3652
# https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
# https://github.com/SangHoon-Joo/Breakout_DQN/blob/master/breakout_test.py
# 코드를 참조했습니다.
import tensorflow as tf
import gym
import itertools
import os
import numpy as np
import random
from tensorflow.contrib.layers import xavier_initializer
from  skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque, namedtuple
from scipy.misc import imresize
from skimage.color import rgb2gray
from skimage.transform import resize

env = gym.make('BreakoutDeterministic-v4')

tf.set_random_seed(777)

IMGSIZE = 84
VALID_ACTIONS = [0, 1, 2, 3]
learning_rate = 0.005
dis = .99
MAX_EPISODE = 500000
BATCH_SIZE = 28
REPLAY_MEMORY = 50000

class DQN:

    def __init__(self, name, summaries_dir=None):

        self.summary_writer = None
        self.name = name
        self.num_hidden = 256
        self.num_action = 4
        self.size_batch = 32

        self._build_model()

        if summaries_dir:
            summary_dir = os.path.join(summaries_dir, "summaries_{}".format(name))
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            self.summary_writer = tf.summary.FileWriter(summary_dir)



    def _build_model(self):

        with tf.variable_scope(self.name) as scope:

            # The target value
            self.Y = tf.placeholder(tf.float32, [None] )
            # Index of selected action
            self.action = tf.placeholder(tf.uint8, [None] )
            # Frame input 84x84x4
            self.X = tf.placeholder(tf.uint8, [None, IMGSIZE, IMGSIZE, 4] )
            self.frames = tf.to_float(self.X)/255.0

            self.f1 = tf.get_variable(name='conv_w1', shape=[8,8,4,16], initializer=xavier_initializer())
            self.f2 = tf.get_variable(name='conv_w2', shape=[4,4,16,32], initializer=xavier_initializer())
            self.w1 = tf.get_variable(name='fc_w1',shape=[2592 ,self.num_hidden], initializer=xavier_initializer())
            self.w2 = tf.get_variable(name='fc_w2', shape=[self.num_hidden ,self.num_action], initializer=xavier_initializer())

            h1 = tf.nn.conv2d(self.frames, self.f1, strides=[1,4,4,1], padding='VALID')
            h1 = tf.nn.relu(h1)
            h2 = tf.nn.conv2d(h1, self.f2, strides=[1,2,2,1], padding='VALID')
            h2 = tf.nn.relu(h2)
            # input의 형태가 [batch_size, ....]일때
            # input을 [batch_size, -1] 로 변환시켜 준다.
            h2 = tf.contrib.layers.flatten(h2)
            fc1 = tf.matmul(h2, self.w1)
            fc1 = tf.nn.relu(fc1)
            # out 변수는 DQN을 거친 결과로 [batch_size, num_action]
            self.pred_qval= tf.matmul(fc1, self.w2)

            # Get the predictions for the chosen actions only
            mask = tf.one_hot(self.action, 4, on_value=1, off_value=0)
            mask = tf.to_float(mask)

            pred_selected_qval = tf.multiply(self.pred_qval, mask)
            self.pred_action = tf.reduce_sum(pred_selected_qval, reduction_indices=[1])

            # Get loss
            self.losses = tf.squared_difference(self.Y, self.pred_action)
            self.loss = tf.reduce_mean(self.losses)

            # Optimalizer
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, momentum=0.95, epsilon=0.01)
            self.train_op = self.optimizer.minimize(self.loss)

            # Tensorboard
            self.summaries = tf.summary.merge([
                tf.summary.scalar("Loss", self.loss),
                tf.summary.scalar("Average Q Var", tf.reduce_mean(self.pred_action))
            ])

    def predict(self, sess, state):

        return sess.run(self.pred_qval, feed_dict={self.X : history_reshape(state)})

    def update(self, sess, state, action, y):

        feed_dict = {self.X : state, self.action : action, self.Y : y}

        summaries, _, loss = sess.run([self.summaries, self.train_op, self.loss], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries)

        return loss

    def get_qVal(self, sess, action, state):
        return sess.run(self.pred_action, feed_dict={self.X : history_reshape(state), self.action : action})


def get_copy_var_ops(*, sess, dest_scope_name="target", src_scope_name="main"):

    op_holder=[]
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dst_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dst_var in zip(src_vars, dst_vars):
        op_holder.append(dst_var.assign(src_var.value()))

    sess.run(op_holder)

def crop_image(image, height_range=(34,195)):
    h_begin, h_end = height_range
    return image[h_begin:h_end, ...]

def resize_image(image, HW_range):
    return imresize(image, HW_range, interp="nearest")

def make_gray_image(image):
    return rgb2gray(image)

def pre_proc(image):
    temp_image = crop_image(image)
    temp_image = make_gray_image(temp_image)
    final_image = resize_image(temp_image, (84, 84))
    return final_image


def pre_process(X):
    x = np.uint8(resize(rgb2gray(X), (84, 84), mode='reflect') * 255)
    return x

def history_init(history, state):

    for i in range(4):
        history[:, :, i] = pre_process(state)

    return history

def history_update(history, new_state):

    for i in range(3):
        history[:, :,i] = history[:, :, i+1]

    history[:, :, -1] = new_state

    return history



def history_reshape(history):
    return np.reshape(history, [-1,84,84,4])


def simple_replay_train(sess, mainDQN, targetDQN, train_batch):

    state_array = np.array([x[0] for x in train_batch])
    action_array = np.array([x[1] for x in train_batch])
    reward_array = np.array([x[2] for x in train_batch])
    next_state_array = np.array([x[3] for x in train_batch])
    done_array = np.array([x[4] for x in train_batch])

    X_batch = state_array
    Y_batch = mainDQN.get_qVal(sess, action_array, state_array)

    for i in range(len(train_batch)):

        if done_array[i] == True:
            Y_batch[i] = reward_array[i]
        else:
            Y_batch[i] = reward_array[i] + dis * np.max(targetDQN.predict(sess, next_state_array[i]))

    loss = mainDQN.update(sess, X_batch, action_array, Y_batch)

    return loss

TRAIN_START = 1000
FINAL_EXPLORATION = 0.1
TARGET_UPDATE = 10000

def main():

    history = np.zeros([84, 84, 4], dtype=np.uint8)
    history_next = np.zeros([84,84,4], dtype=np.uint8)
    replay_buffer = deque(maxlen=REPLAY_MEMORY)
    last_100_game_reward = deque(maxlen=100)


    mainDQN = DQN("main")
    targetDQN = DQN("target")

    SAVER_DIR = "./save/"
    saver = tf.train.Saver()
    checkpoint_path = os.path.join(SAVER_DIR, "model")
    ckpt = tf.train.get_checkpoint_state(SAVER_DIR)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        if ckpt and ckpt.model_checkpoint_path:
            print("[Restore Model]")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        get_copy_var_ops(sess=sess, dest_scope_name="target", src_scope_name="main")

        e = 1.
        frame = 0

        for i in range(MAX_EPISODE):

            done = False
            state = env.reset()
            history = history_init(history, state)

            step_count = 0

            while not done:

                frame += 1

                if e > FINAL_EXPLORATION and frame > TRAIN_START:
                    e -= (1. - FINAL_EXPLORATION)/ 500000

                if np.random.rand(1) < e:
                    action = np.random.randint(4)
                else:
                    action = np.argmax(mainDQN.predict(sess, history))

                next_state, reward, done, info = env.step(action)
                reward = np.clip(reward, -1, 1)

                next_state = pre_process(next_state)


                replay_buffer.append((history, action, reward, next_state, done))

                if len(replay_buffer) > REPLAY_MEMORY:
                    replay_buffer.popleft()

                history = history_update(history, next_state)

                step_count += 1

                env.render()


            if frame > TRAIN_START:

                minibatch = random.sample(replay_buffer, 32)
                loss = simple_replay_train(sess, mainDQN, targetDQN, minibatch)

                if i % 100 == 0:
                    saver.save(sess, checkpoint_path)
                    print("Episode: {}, Loss: {}".format(i, loss))

                if frame % 10000 == 0:
                    get_copy_var_ops(sess=sess, dest_scope_name="target", src_scope_name="main")




def bot_play(sess, mainDQN, env=env):
    history = np.zeros([84, 84, 4], dtype=np.uint8)
    state = env.reset()
    reward_sum = 0
    x = pre_proc(state)
    history = history_init(history, x)
    env.step(1)
    while True:
        env.render()

        Qs = mainDQN.predict(sess, history)
        action = np.argmax(Qs)

        state, reward, done, info = env.step(action)
        reward_sum += reward
        x = pre_proc(state)
        history = history_update(history, x)

        if info['ale.lives'] < 5:
            done = True

        if done:
            print("Total score: {}".format(reward_sum))
            break

if __name__=="__main__":
    main()