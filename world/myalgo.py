
import numpy as np
import threading
import time
from datetime import datetime
import jderobot
import math
import cv2
from math import pi as pi
import os, errno
import sys



time_cycle = 5
import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf
import time
import random
from collections import deque
from Env import Gazeboworld
# from actorcritic import ActorCritic as a
from ac import ActorCritic

class MyAlgorithm(threading.Thread):

	def __init__(self, args):
		self.stop_event = threading.Event()
		self.kill_event = threading.Event()
		self.lock = threading.Lock()
		self.args = args
		threading.Thread.__init__(self, args=self.stop_event)


	def run (self):

		while (not self.kill_event.is_set()):
		   
			start_time = datetime.now()

			if not self.stop_event.is_set():
				self.execute()

			finish_Time = datetime.now()

			dt = finish_Time - start_time
			ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
			#print (ms)
			if (ms < time_cycle):
				time.sleep((time_cycle - ms) / 1000.0)


	def stop (self):
		self.stop_event.set()


	def play (self):
		if self.is_alive():
			self.stop_event.clear()
		else:
			self.start()


	def kill (self):
		self.kill_event.set()
			
		 
	def execute(self):
		sess = tf.Session()
		K.set_session(sess)
		env = Gazeboworld()
		actor_critic = ActorCritic(env)
		t = time.time()
		value = datetime.fromtimestamp(t)
		t_str = value.strftime('%m_%d_%H_%M')
		dir_p = t_str+'_weights'
		dir_a = t_str+'_weights/actor'
		dir_c = t_str+'_weights/critic'
		try:
			os.makedirs(dir_p)
			os.makedirs(dir_a)
			os.makedirs(dir_c)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

		if len(self.args) > 2:
			actor_critic.load_trained_model(self.args[1], self.args[2])
			# sys.exit(0)

		def saveWeights(actor_critic, episode, dir_a, dir_c):
			weights = actor_critic.target_actor_model.save_weights(dir_a+'/w_E{}.h5'.format(episode))
			weights = actor_critic.target_critic_model.save_weights(dir_c+'/w_E{}.h5'.format(episode))
			print 'Weights Saved For Episode: {}'.format(episode)

		num_trails = 10000
		trial_len  = 5000
		for i in xrange(num_trails):
			cur_state = env.reset()
			done = False
			while not done:
				cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
				action = actor_critic.act(cur_state)
				action = action.reshape((1, env.action_space.shape[0]))
				new_state, reward, done = env.step(action)
				# new_state, reward, done = env.step([[1,1]])
				try:
					new_state, reward, done = env.step(action)
				except Exception as e:
					print e
					continue
				new_state = new_state.reshape((1, env.observation_space.shape[0]))
				actor_critic.remember(cur_state, action, reward, new_state, done)
				actor_critic.train()
				# print env.isDead(), done, min(env.laser.getLaserData().distanceData[:360])
				if done:
					env.reset()
					print "Done....<<<<<<<<<<<<<<<<<<... ", env.death
				# cur_state = np.array(env.state)
				sys.stdout.flush()
			print "Length of memory: {}".format(len(actor_critic.memory))
			actor_critic.epsilon *= actor_critic.epsilon_decay
			print "EPSILON =================================> {}".format(actor_critic.epsilon)
			saveWeights(actor_critic, i, dir_a, dir_c)

if __name__ == '__main__':
	args = sys.argv
	a = MyAlgorithm(args)
	a.start()