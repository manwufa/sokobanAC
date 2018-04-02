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
from collections import deque
myLib=np.ctypeslib.load_library('libsokoban', '/home/wf/CLionProjects/sokoban/cmake-build-release/')
GameNew=myLib.py_sokoban
GameMov=myLib.py_move
GameBNew=myLib.py_newgame_batch
GameBMov=myLib.py_move_batch

pfloat=np.ctypeslib.ndpointer(dtype = np.float32)
pint=np.ctypeslib.ndpointer(dtype = np.int32)
GameNew.argtypes=[pfloat,pfloat,pfloat,pfloat,pfloat,pfloat,c_int32,c_int32]
GameMov.argtypes=[pfloat,pfloat,pfloat,pfloat,pfloat,c_int32,pfloat]
GameBNew.argtypes=[pfloat,pfloat,pfloat,pfloat,c_int32,c_int32,pint]
GameBMov.argtypes=[pfloat,pfloat,pfloat,pint,pfloat,c_int32]


#two extern funtion from c
#myLib = cdll.LoadLibrary("/home/wf/CLionProjects/sokoban/cmake-build-release/libsokoban.so")
gridSz = 8
YEX=20
batchSz = 32

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
        self.state = np.zeros((batchSz,gridSz*gridSz*4), dtype="float32")
        self.valiMov = np.zeros((batchSz,4), dtype="float32")
        self.curScore = np.zeros(batchSz, dtype="float32")
        self.statefinish = np.ones((batchSz), dtype="float32")
        self.move=np.zeros((batchSz), dtype="float32")
        self.numBox=numBox


    def new_episode(self):
        random.seed()
        #seed=np.random.randint(low=0,high=65535, size=batchSz)
        seed=np.ones((batchSz), dtype="int32")
        #py_newgame_batch(float * outbuff, float * validA, float* rightbox,float * updateTag, int  batchSize, int  numBox, int * seed)
        GameBNew(self.state,self.valiMov,self.curScore,self.statefinish,batchSz,4,seed);
        for i in range(batchSz):
            if self.statefinish[i]==1:
                self.statefinish[i]=0
                self.move[i]=0
    def make_action(self,a):
        outstate = np.zeros((batchSz,gridSz*gridSz*4), dtype="float32")
        prevscore= self.curScore.copy()
        #py_move_batch(float* inbuff,float* outbuff,float *validA,int* a,float* rightBox,int batchSize)
        GameBMov(self.state,outstate,self.valiMov,a,self.curScore,batchSz)
        self.state=outstate
        reward=(self.curScore-prevscore)
        for i in range(batchSz):
            if self.curScore[i]==self.numBox:
                reward[i]=10
                self.statefinish[i]=1
            if self.move[i] > 50:
                self.statefinish[i] = 1
                reward[i] = -1
        self.move+=1
        return reward

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
            self.Q = slim.fully_connected(hidden, a_size,
                                               activation_fn=None,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=tf.constant_initializer(0.0))
            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.actions = tf.placeholder(shape=[None] ,dtype=tf.int32)
                self.actions_onehot = tf.one_hot(self.actions ,a_size ,dtype=tf.float32)
                self.target_Q = tf.placeholder(shape=[None] ,dtype=tf.float32)
                self.responsible_outputs = tf.reduce_sum(self.Q * self.actions_onehot, [1])
                self.loss = tf.reduce_mean(tf.square(self.target_Q - self.responsible_outputs))

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss ,local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads ,self.grad_norms = tf.clip_by_global_norm(self.gradients ,400000.0)

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
        self.env = game

    def work(self ,max_episode_length ,gamma ,sess ,coord ,saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print ("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            buffer = deque()
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_frames = []
                while len(buffer)>100:
                    buffer.popleft()

                episode_reward = 0
                episode_step_count = 0
                d = False
                self.env.new_episode()
                while self.env.statefinish[0] == False:
                    self.env.new_episode()
                    s = self.env.state.copy()
                    # Take an action using probabilities from policy network output.
                    a_dist= sess.run(self.local_AC.Q ,feed_dict={self.local_AC.inputs :s})
                    valid_move=self.env.valiMov.copy()
                    a_dist=valid_move*(np.asarray(a_dist)+np.random.random_sample(valid_move.shape)+1000)-1000
                    epsgreed = np.random.random_sample(batchSz)<0.9
                    a = (np.argmax(a_dist, axis=1)*epsgreed+
                        np.argmax(valid_move*(np.random.random_sample(valid_move.shape)+1),axis=1)*(1-epsgreed)).astype(np.int32)
                    r = self.env.make_action(a)
                    d = self.env.statefinish.copy()
                    s1 = self.env.state.copy()
                    episode_frames.append(toframe(s1[0,:]))
                    buffer.append([s ,a ,r ,s1 ,d])

                    episode_reward += r
                    total_steps += 1
                    episode_step_count += 1

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                print (episode_reward)
                qq=len(buffer)
                g_n=0
                v_n=0
                while (qq>0):
                    qq-=1
                    Q = np.amax(sess.run(self.local_AC.Q, feed_dict={self.local_AC.inputs: buffer[qq][3]}),axis=1)
                    Q=buffer[qq][2]+Q*gamma*(1.0-buffer[qq][4])
                    Q2,g_n, v_n, _ = sess.run(
                        [self.local_AC.Q,self.local_AC.grad_norms, self.local_AC.var_norms, self.local_AC.apply_grads],
                        feed_dict={self.local_AC.target_Q: Q, self.local_AC.inputs: buffer[qq][0],
                                   self.local_AC.actions: buffer[qq][1]})
                    sess.run(self.update_local_ops)
                    q2a=np.amax(Q2,axis=1)
                    Q3 = np.amax(sess.run(self.local_AC.Q, feed_dict={self.local_AC.inputs: buffer[qq][0]}), axis=1)
                    print('q2a=',sum(q2a),'q3=',sum(Q3),'Q=',sum(Q))
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
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
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
    trainer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
    master_network = AC_Network(s_size ,a_size ,'global' ,None) # Generate global network
    num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
    num_workers=1
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