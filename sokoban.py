import threading
import multiprocessing
import random
import numpy as np
from numpy.ctypeslib import ndpointer
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
from time import sleep
import scipy.signal
from helper import *
from ctypes import c_float
from ctypes import c_int32
from ctypes import cdll

myLib=np.ctypeslib.load_library('libsokoban', '/home/wf/CLionProjects/sokoban/cmake-build-release/')
GameNew=myLib.py_sokoban
GameMov=myLib.py_move
pfloat=np.ctypeslib.ndpointer(dtype = np.float32)
GameNew.argtypes=[pfloat,pfloat,pfloat,pfloat,pfloat,pfloat,c_int32,c_int32]
GameMov.argtypes=[pfloat,pfloat,pfloat,pfloat,pfloat,c_int32,pfloat]
#two extern funtion from c
#myLib = cdll.LoadLibrary("/home/wf/CLionProjects/sokoban/cmake-build-release/libsokoban.so")
gridSz = 8
YEX=20

def update_target_graph(from_scope ,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var ,to_var in zip(from_vars ,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder
#ret=x[i]+gamma*x[i+1]
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer
def toframe(s):
    img1 = np.zeros((gridSz * YEX, gridSz * YEX, 3), dtype="uint8")
    for h in range(gridSz):
        for w in range(gridSz):
            if s[h*4*gridSz+w*4] == 0:
                img1[h * YEX:(h + 1) * YEX, w * YEX:(w + 1) * YEX] = 64, 64, 64
            elif s[h*4*gridSz+w*4+1] == 0:
                img1[h * YEX:(h + 1) * YEX, w * YEX:(w + 1) * YEX] = 128, 128, 255
            elif s[h*4*gridSz+w*4+2] == 0:
                img1[h * YEX:(h + 1) * YEX, w * YEX:(w + 1) * YEX] = 192, 192, 0
            else:
                img1[h * YEX:(h + 1) * YEX, w * YEX:(w + 1) * YEX] = 0, 192, 0
            if s[h*4*gridSz+w*4+3] == 1:
                img1[h * YEX + 3:h * YEX + YEX-3, w * YEX + 3:w * YEX + YEX-3] = 128, 0, 128
            if s[h*4*gridSz+w*4+2] == 1:
                img1[h * YEX + int(YEX/2)-1:h * YEX + int(YEX/2)+1, w * YEX + 3:w * YEX + YEX-3] = 128, 0, 0
                img1[h * YEX + 3:h * YEX + YEX-3, w * YEX + int(YEX/2)-1:w * YEX + int(YEX/2)+1] = 128, 0, 0
    return img1


class sokobanGame():
    def __init__(self ,numBox=4):
        self.grid = np.zeros((gridSz*gridSz), dtype="float32")
        self.box = np.zeros((gridSz*gridSz), dtype="float32")
        self.boxGoal = np.zeros((gridSz*gridSz), dtype="float32")
        self.playerPos = np.zeros((gridSz*gridSz), dtype="float32")
        self.playerPosGoal = np.zeros((gridSz*gridSz), dtype="float32")
        self.vailMov = np.zeros(4, dtype="float32")
        self.numBox=numBox

    def new_episode(self):
        random.seed()
        #myLib.py_sokoban(self.grid, self.box, self.boxGoal, self.playerPos, self.playerPosGoal, self.numBox, random.randint(0, 65535))
        myLib.py_sokoban(self.grid, self.box, self.boxGoal, self.playerPos, self.playerPosGoal,self.vailMov, self.numBox,1)
        a1=np.zeros((gridSz*gridSz), dtype="float32")
        a2 = np.zeros((gridSz*gridSz), dtype="float32")
        a1[:]=self.box
        a2[:]=self.boxGoal
        self.curScore = sum(a1 * a2)
        self.move=0
    def get_state(self):
        s=np.zeros((gridSz*gridSz, 4), dtype="float32")
        s[:,0]=self.grid
        s[:,1]=self.box
        s[:,2]=self.boxGoal
        s[:,3]=self.playerPos
        s = np.reshape(s, [np.prod(s.shape)])
        return s
    def is_episode_finished(self):
        return self.move>50 or self.curScore==self.numBox
    def make_action(self,a):
        box2 = np.zeros((gridSz*gridSz), dtype="float32")
        playerPos2 = np.zeros((gridSz*gridSz), dtype="float32")
        #s1=self.get_state()
        myLib.py_move(self.grid, self.box, self.playerPos, box2, playerPos2, int(a),self.vailMov)
        #s1 = np.zeros((gridSz * gridSz), dtype="float32")
        #s2 = np.zeros((gridSz * gridSz), dtype="float32")
        #s1[:] = self.playerPos
        #s2[:] = playerPos2
        self.box = box2
        self.playerPos = playerPos2
        #s2=self.get_state()
        prevScore=self.curScore
        a1=np.zeros((gridSz*gridSz), dtype="float32")
        a2 = np.zeros((gridSz*gridSz), dtype="float32")
        a1[:]=self.box
        a2[:]=self.boxGoal
        self.curScore = sum(a1 * a2)
        self.move+=1
        rew=10 if self.curScore==self.numBox else 1*(self.curScore-prevScore)
        return rew
    def get_valid_move(self):
        a=np.zeros(4,dtype="float32")
        a[:]=self.vailMov
        return a






#different from doom setting this game is pure MDP that need no LSTM
class AC_Network():
    def __init__(self ,s_size ,a_size ,scope ,trainer):
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.inputs = tf.placeholder(shape=[None ,s_size] ,dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs ,shape=[-1 ,8 ,8 ,4])
            self.conv1 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.imageIn ,num_outputs=32,
                                     kernel_size=[3 ,3] ,stride=[1 ,1] ,padding='SAME')
            self.conv1a = slim.conv2d(activation_fn=tf.nn.elu,
                                      inputs=self.conv1, num_outputs=64,
                                     kernel_size=[3, 3], stride=[1, 1], padding='SAME')
            self.conv1b = slim.conv2d(activation_fn=tf.nn.elu,
                                      inputs=self.conv1a, num_outputs=64,
                                     kernel_size=[3, 3], stride=[1, 1], padding='SAME')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.elu,
                                     inputs=self.conv1b ,num_outputs=32,
                                     kernel_size=[3 ,3] ,stride=[1 ,1] ,padding='SAME')
            hidden1 = slim.fully_connected(slim.flatten(self.conv2) ,512 ,activation_fn=tf.nn.elu)
            hidden = slim.fully_connected(hidden1, 512, activation_fn=tf.nn.elu)
            self.policy = slim.fully_connected(hidden, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(hidden, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)
            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None] ,dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions ,a_size ,dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None] ,dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None] ,dtype=tf.float32)
                self.actions_fact = tf.placeholder(shape=[None,4] ,dtype=tf.float32)
                self.actions_mask = tf.placeholder(shape=[None, 4], dtype=tf.float32)
                self.mask_policy=self.policy *self.actions_mask
                self.mask_policy_sum=tf.reduce_sum(self.mask_policy,[1])
                self.actual_policy = self.mask_policy/tf.reshape(self.mask_policy_sum,[-1,1])
                self.responsible_outputs = tf.reduce_sum(self.actual_policy * self.actions_onehot, [1])

                # Loss functions
                #self.policysum_loss= tf.reduce_sum(tf.square(self.mask_policy_sum-0.6-0.1*(tf.reduce_sum(self.actions_mask,[1]))))
                self.policysum_loss=tf.reduce_sum(tf.square((self.policy-0.01) *(1.0-self.actions_mask)))

                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value ,[-1])))
                self.entropy = - tf.reduce_sum(self.actual_policy * tf.log(self.actual_policy+0.0000000001))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs ) *self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.05+self.policysum_loss*0.05

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss ,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads ,self.grad_norms = tf.clip_by_global_norm(self.gradients ,40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads ,global_vars))

class Worker():
    def __init__(self ,game ,name ,s_size ,a_size ,trainer ,model_path ,global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_ " +str(self.number))

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(s_size ,a_size ,self.name ,trainer)
        self.update_local_ops = update_target_graph('global' ,self.name)

        # The Below code is related to setting up the Doom environment
        self.actions = self.actions = np.identity(a_size ,dtype=bool).tolist()
        # End Doom set-up
        self.env = game

    def train(self ,rollout ,sess ,gamma ,bootstrap_value):

        rollout = np.array(rollout)
        observations = rollout[: ,0]
        actions = rollout[: ,1]
        rewards = rollout[: ,2]
        next_observations = rollout[: ,3]
        values = rollout[: ,5]
        actions_mask = rollout[:, 6]


        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus ,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages ,gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v :discounted_rewards,
                     self.local_AC.inputs :np.vstack(observations),
                     self.local_AC.actions :actions,
                     self.local_AC.advantages :advantages,
                     self.local_AC.actions_mask: np.vstack(actions_mask)}
        v_l ,p_l ,e_l ,g_n ,v_n,_ = sess.run([self.local_AC.value_loss,
                                            self.local_AC.policy_loss,
                                            self.local_AC.entropy,
                                            self.local_AC.grad_norms,
                                            self.local_AC.var_norms,
                                            self.local_AC.apply_grads],
                                           feed_dict=feed_dict)
        return v_l / len(rollout) ,p_l / len(rollout) ,e_l / len(rollout), g_n ,v_n

    def work(self ,max_episode_length ,gamma ,sess ,coord ,saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                self.env.new_episode()
                s = self.env.get_state()
                episode_frames.append(toframe(s))
                while self.env.is_episode_finished() == False:
                    # Take an action using probabilities from policy network output.
                    a_dist ,v  = sess.run([self.local_AC.policy ,self.local_AC.value ],
                                                  feed_dict={self.local_AC.inputs :[s]})
                    valid_move=self.env.get_valid_move()
                    #valid_move[:]=1
                    a1=a_dist[0]*valid_move
                    a1sum=a1.sum()
                    a1/=a1sum



                    a = np.random.choice(a1 ,p=a1)
                    a = np.argmax(a1 == a)

                    r = self.env.make_action(a)
                    d = self.env.is_episode_finished()
                    s1 = self.env.get_state()
                    episode_frames.append(toframe(s1))
                    episode_buffer.append([s ,a ,r ,s1 ,d ,v[0 ,0],valid_move])
                    episode_values.append(v[0 ,0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == 10 and d != True and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value,feed_dict={self.local_AC.inputs :[s]})[0 ,0]
                        v_l ,p_l ,e_l ,g_n ,v_n = self.train(episode_buffer ,sess ,gamma ,v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))
                print (episode_reward)
                qq=len(episode_buffer)
                while (qq>0):
                    qq2=max(qq-10,0)
                    bs=0.0
                    if qq!=len(episode_buffer):
                        bs=sess.run(self.local_AC.value,feed_dict={self.local_AC.inputs :[episode_buffer[qq][0]]})[0 ,0]
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer[qq2:qq], sess, gamma, bs)
                    sess.run(self.update_local_ops)
                    qq=qq2

                # Update the network using the episode buffer at the end of the episode.
                #if len(episode_buffer) != 0:
                #    if len(episode_buffer)<100 :
                #        v_l ,p_l ,e_l ,g_n ,v_n = self.train(episode_buffer ,sess ,gamma ,0.0 )
                #    else:
                        #v1 = sess.run(self.local_AC.value, feed_dict={self.local_AC.inputs: [s]})[0, 0]
                #        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)


                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        make_gif(images ,'./frames/image ' +str(episode_count ) +'.gif',
                                 duration=len(images ) *time_per_step ,true_image=True ,salience=False)
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess ,self.model_path +'/model- ' +str(episode_count ) +'.cptk')
                        print ("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1

max_episode_length = 300
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = 256 # Observations are greyscale frames of 84 * 84 * 1
a_size = 4 # Agent can move Left, Right, or Fire
load_model = False
model_path = './model'

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0 ,dtype=tf.int32 ,name='global_episodes' ,trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-5)
    master_network = AC_Network(s_size ,a_size ,'global' ,None) # Generate global network
    num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(Worker(sokobanGame() ,i ,s_size ,a_size ,trainer ,model_path ,global_episodes))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    #load_model = True
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess ,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # This is where the asynchronous magic happens.
    # Start the "work" process for each worker in a separate threat.
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(max_episode_length ,gamma ,sess ,coord ,saver)
        t = threading.Thread(target=(worker_work))
        t.start()
        sleep(0.5)
        worker_threads.append(t)
    coord.join(worker_threads)