
import numpy as np
import threading
import time
from datetime import datetime
import jderobot
import math
import cv2
from math import pi as pi

time_cycle = 80
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
from actorcritic import ActorCritic

class MyAlgorithm(threading.Thread):

	def __init__(self):
		self.stop_event = threading.Event()
		self.kill_event = threading.Event()
		self.lock = threading.Lock()
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
		actor_critic = ActorCritic(env, sess)

		def saveWeights(actor_critic, episode):
			weights = actor_critic.target_actor_model.save_weights('weight_{}.h5'.format(episode))
			print 'Weights Saved For Episode: {}'.format(episode)

		num_trails = 10000
		trial_len  = 5000
		for i in xrange(num_trails):
			cur_state = env.reset()
			action = env.randomAction()
			for t_step in xrange(trial_len):
				t = time.time()
				cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
				action = actor_critic.act(cur_state)
				action = action.reshape((1, env.action_space.shape[0]))
				try:
					new_state, reward, done = env.step(action)
				except Exception as e:
					print e
					continue
				print "----------------------------------"
				new_state = new_state.reshape((1, env.observation_space.shape[0]))
				actor_critic.remember(cur_state, action, reward, new_state, done)
				actor_critic.train()
				if done:
					env.halt()
					break
				cur_state = new_state
			saveWeights(actor_critic, i)

if __name__ == '__main__':
	a = MyAlgorithm()
	a.start()