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
            self.X = tf.placeholder(tf.float32, [None, IMGSIZE, IMGSIZE, 4] )

            self.f1 = tf.get_variable(name='conv_w1', shape=[8,8,4,16], initializer=xavier_initializer())
            self.f2 = tf.get_variable(name='conv_w2', shape=[4,4,16,32], initializer=xavier_initializer())
            self.w1 = tf.get_variable(name='fc_w1',shape=[2592 ,self.num_hidden], initializer=xavier_initializer())
            self.w2 = tf.get_variable(name='fc_w2', shape=[self.num_hidden ,self.num_action], initializer=xavier_initializer())

            h1 = tf.nn.conv2d(self.X, self.f1, strides=[1,4,4,1], padding='VALID')
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
            self.pred_action = tf.reduce_sum(pred_selected_qval, reduction_indices=1)

            # Get loss
            #self.losses = tf.squared_difference(self.Y, self.pred_action)
            #self.loss = tf.reduce_mean(self.losses)

            self.loss = tf.losses.huber_loss(self.Y, self.pred_action)

            # Optimalizer
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, momentum=0.95, epsilon=0.01)
            self.train_op = self.optimizer.minimize(self.loss)

            # Tensorboard
            self.summaries = tf.summary.merge([
                tf.summary.scalar("Loss", self.loss),
                tf.summary.scalar("Average Q Var", tf.reduce_mean(self.pred_action))
            ])

    def predict(self, sess, state):

        return sess.run(self.pred_qval, feed_dict={self.X : np.reshape(state, [-1, 84, 84, 4])})

    def update(self, sess, state, action, y):

        feed_dict = {self.X : state, self.action : action, self.Y : y}

        summaries, _, loss = sess.run([self.summaries, self.train_op, self.loss], feed_dict)

        if self.summary_writer:
            self.summary_writer.add_summary(summaries)

        return loss

    def get_qVal(self, sess, action, state):
        return sess.run(self.pred_action, feed_dict={self.X : np.reshape(state, [-1, 84, 84, 4]), self.action : action})


    def get_action(self, qVal, e):
        if e > np.random.rand(1):
            action = np.random.randint(4)
        else:
            action = np.argmax(qVal)
        return action

def get_copy_var_ops(*, sess, dest_scope_name="target", src_scope_name="main"):

    op_holder=[]
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dst_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dst_var in zip(src_vars, dst_vars):
        op_holder.append(dst_var.assign(src_var.value()))

    sess.run(op_holder)


def pre_process(state):
    x = np.uint8(resize(rgb2gray(state), (84, 84), mode='reflect') * 255)
    return x

def history_init(history, state):

    for i in range(4):
        history[:, :, i] = pre_process(state)


def simple_replay_train(sess, mainDQN, targetDQN, mini_batch):

    #print("[Bef]", np.shape(mini_batch))
    mini_batch = np.array(mini_batch).transpose()
    #print("[Aft]", np.shape(mini_batch))

    history = np.stack(mini_batch[0], axis=0)
    #print("[Hist]", np.shape(history))

    states = np.float32(history[:, :, :, :4]) / 255.
    actions = list(mini_batch[1])
    rewards = list(mini_batch[2])
    next_states = np.float32(history[:, :, :, 1:]) / 255.
    dones = mini_batch[3]

    # bool to binary
    dones = dones.astype(int)

    Q1 = targetDQN.predict(sess, next_states)
    #print("Q1::", Q1)
    #print("Max::", np.max(Q1, axis=1))

    y = rewards + (1 - dones) * 0.95 * np.max(Q1, axis=1)

    # 업데이트 된 Q값으로 main네트워크를 학습
    loss = mainDQN.update(sess, states, actions, y)

    return loss

TRAIN_START = 1000
FINAL_EXPLORATION = 0.1
TARGET_UPDATE = 10000
IMGSIZE = 84
learning_rate = 0.005
MAX_EPISODE = 500000
BATCH_SIZE = 32
REPLAY_MEMORY = 100000

def main():


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
        replay_buffer = deque(maxlen=REPLAY_MEMORY)

        averageQ = deque()

        env.reset()
        _,_,_, info = env.step(0)
        life = info['ale.lives']

        for i in range(MAX_EPISODE):


            history = np.zeros([84, 84, 5], dtype=np.uint8)
            done = False
            count = 0
            state = env.reset()
            history_init(history, state)


            while not done:

                frame += 1
                count += 1

                if e > FINAL_EXPLORATION and frame > TRAIN_START:
                    e -= (1. - FINAL_EXPLORATION)/ 1000000

                Q = mainDQN.predict(sess, np.float32(history[:,:,:4])/255.)
                averageQ.append(np.max(Q))
                action = mainDQN.get_action(Q, e)

                next_state, reward, done, info = env.step(action)
                reward = np.clip(reward, -1, 1)

                if life > info['ale.lives']:
                    ter = True;
                else:
                    ter = False;

                history[:,:,4] = pre_process(next_state)


                replay_buffer.append((np.copy(history[:,:,:]), action, reward, ter))

                history[:, :,:4] = history[:,:,1:]

                #env.render()

                if frame > TRAIN_START:

                    minibatch = random.sample(replay_buffer, 32)
                    loss = simple_replay_train(sess, mainDQN, targetDQN, minibatch)

                    if i % 1000 == 0:
                        saver.save(sess, checkpoint_path)
                        #print("\nEpisode: {}, Loss: {}\n".format(i, loss))

                    if frame % 50000 == 0:
                        averageQ = deque()

                    if frame % 1000 == 0:
                        get_copy_var_ops(sess=sess, dest_scope_name="target", src_scope_name="main")


            print("Episode {0:6d} | PlayCount {1:5d} | e-greedy:{2:.5f} | Average Q {3:2.5f}".format(i, count, e, np.mean(averageQ)))




if __name__=="__main__":
    main()