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
from collections import deque, namedtuple
#from lib import plotting

env = gym.make('BreakoutDeterministic-v4')
IMGSIZE = 84
VALID_ACTIONS = [0, 1, 2, 3]


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


def get_copy_var_ops(*, sess, dest_scope_name="target", src_scope_name="main"):

    op_holder=[]
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dst_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dst_vars):
        op_holder.append(dst_vars.assign(src_var.value()))

    sess.run(op_holder)


class DQN:

    def __init__(self, name, summaries_dir=None):

        self.summary_writer = None
        self.name = name
        self.num_hidden = 256
        self.num_action = 4
        self.size_batch = 32

        self.forward()

        if summaries_dir:
            summary_dir = os.path.join(summaries_dir, "summaries_{}".format(name))
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            self.summary_writer = tf.summary.FileWriter(summary_dir)



    def forward(self):

        with tf.variable_scope(self.name):

            # The target value
            self.Y = tf.placeholder(tf.float32, [self.size_batch] )
            # Index of selected action
            self.action = tf.placeholder(tf.uint8, [self.size_batch] )
            # Frame input 84x84x4
            self.X = tf.placeholder(tf.uint8, [self.size_batch, IMGSIZE, IMGSIZE, 4] )
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
            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.95,0.0,1e-6)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

            # Tensorboard
            self.summaries = tf.summary.merge([
                tf.summary.scalar("Loss", self.loss),
                tf.summary.scalar("Average Q Var", tf.reduce_mean(self.pred_action))
            ])

    def pred(self, sess, state):
        return sess.run(self.pred_qval, feed_dict={self.X : state})

    def update(self, sess, state, action, y):
        feed_dict = {self.X : state, self.action :action, self.Y : y}
        summaries, step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step() ,self.train_op, self.loss], feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries)
        return loss


def make_epsilon_greedy_policy( model , num_action ):

    def policy_fn(sess, state , epsilon):
        temp = np.ones(num_action, dtype=float) * epsilon / num_action

        q_values = model.pred(sess, np.expand_dims(state, 0))[0]
        best_action = np.argmax(q_values)

        temp[best_action] += (1.0 - epsilon)
        # 확률 값의 array를 반환해주도록 하는 함수
        return temp
    return policy_fn


def deep_q_learning(sess,
                    env,
                    main_model,
                    target_model,

                    state_processor,
                    num_episodes,

                    experiment_dir,

                    replay_memory_size=500000,
                    replay_memory_init_size=50000,
                    update_target_model_every=10000,

                    discount_factor=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    batch_size=32,
                    record_video_every=50):

    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    Args:
        sess: Tensorflow Session object
        env: OpenAI environment
        main_model: Estimator object used for the q values
        target_model: Estimator object used for the targets
        state_processor: A StateProcessor object
        num_episodes: Number of episodes to run for
        experiment_dir: Directory to save Tensorflow summaries in
        replay_memory_size: Size of the replay memory
        replay_memory_init_size: Number of random experiences to sampel when initializing
          the reply memory.
        update_target_model_every: Copy parameters from the Q estimator to the
          target estimator every N steps
        discount_factor: Gamma discount factor
        epsilon_start: Chance to sample a random action when taking an action.
          Epsilon is decayed over time and this is the start value
        epsilon_end: The final minimum value of epsilon after decaying is done
        epsilon_decay_steps: Number of steps to decay epsilon over
        batch_size: Size of batches to sample from the replay memory
        record_video_every: Record a video every N episodes
    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # Replay memory =======================================================
    replay_memory = []
    # =====================================================================


    # Directory Check ====================================================
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)
    # =====================================================================

    # Save Information Check ==============================================
    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
    # =====================================================================

    # Epsilon Gradient ====================================================
    # Epsilon Gradient 를 위해서 학습 과정에서의 Step Information을 받아와야한다.
    # 이를 위해서 optimizer.minimize(global_step=tf.contrib.framework.get_global_step()) 와 같이 선언을 해두었다.
    total_step = sess.run(tf.contrib.framework.get_global_step())

    # Epsilon Decay 방법을 정의 =============================================
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    policy = make_epsilon_greedy_policy(
        main_model,
        4)
    # =====================================================================

    # Replay memory =======================================================
    # Frame 전처리================================
    state = env.reset()
    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)
    # ===========================================

    for i in range(replay_memory_init_size):

        action_probs = policy(sess, state, epsilons[min(total_step, epsilon_decay_steps-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        next_state, reward, done, _ = env.step(VALIDE_ACTIONS[action])

        next_state = state_processor.process(sess, next_state)

        next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

        replay_memory.append(Transition(state, action, reward, next_state, done))

        if done:
            state = env.reset()
            state = state_processor.process(sess, state)
            state = np.stack([state] * 4, axis=2)

        else:
            state = next_state

    # Record videos
    # Use the gym env Monitor wrapper
    env = Monitor(env,
                  directory=monitor_path,
                  resume=True,
                  video_callable=lambda count: count % record_video_every ==0)


    for i_episode in range(num_episodes):

        # Save the current checkpoint
        saver.save(tf.get_default_session(), checkpoint_path)

        # Frame 전처리================================
        state = env.reset()
        state = state_processor.process(sess, state)
        state = np.stack([state] * 4, axis=2)
        loss = None
        # ===========================================

        # t = 1 to T ================================
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_step, epsilon_decay_steps-1)]

            # Add epsilon to Tensorboard
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=epsilon, tag="epsilon")
            main_model.summary_writer.add_summary(episode_summary, total_step)

            # Maybe update the target estimator
            if total_step % update_target_estimator_every == 0:
                get_copy_var_ops(sess, 'target', 'main')
                print("\nCopied model parameters to target network.")

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_step, i_episode + 1, num_episodes, loss), end="")
            sys.stdout.flush()

            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
            next_state = state_processor.process(sess, next_state)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))


            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets (Double DQN)
            q_values_next = main_model.pred(sess, next_states_batch)
            best_actions = np.argmax(q_values_next, axis=1)

            q_values_next_target = target_model.pred(sess, next_states_batch)

            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

            # Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = main_model.update(sess, states_batch, action_batch, targets_batch)

            if done:
                break

            state = next_state
            total_step += 1

        # Add summaries to tensorboard
        episode_summary = tf.Summary()
        q_estimator.summary_writer.add_summary(episode_summary, total_step)
        q_estimator.summary_writer.flush()


    env.monitor.close()
    return stats


tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
main_model = DQN('main', summaries_dir=experiment_dir)
target_model = DQN('target')

# State processor
state_processor = StateProcessor()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in deep_q_learning(sess,
                                    env,
                                    main_model=main_model,
                                    target_model=target_model,
                                    state_processor=state_processor,
                                    experiment_dir=experiment_dir,
                                    num_episodes=10000,
                                    replay_memory_size=500000,
                                    replay_memory_init_size=50000,
                                    update_target_model_every=10000,
                                    epsilon_start=1.0,
                                    epsilon_end=0.1,
                                    epsilon_decay_steps=500000,
                                    discount_factor=0.99,
                                    batch_size=32):
        print("Step[", t, "]")
