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

class ActorCritic:
	def __init__(self, env, sess):
		self.env  = env
		self.sess = sess

		self.learning_rate = 0.001
		self.epsilon = 1.0
		self.epsilon_decay = .9999
		self.gamma = .95
		self.tau   = .125

		# ===================================================================== #
		#                               Actor Model                             #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #

		self.memory = deque(maxlen=7500)
		self.actor_state_input, self.actor_model = self.create_actor_model()
		_, self.target_actor_model = self.create_actor_model()

		self.actor_critic_grad = tf.placeholder(tf.float32, 
			[None, self.env.action_space.shape[0]]) # where we will feed de/dC (from critic)
		
		actor_model_weights = self.actor_model.trainable_weights
		self.actor_grads = tf.gradients(self.actor_model.output, 
			actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
		grads = zip(self.actor_grads, actor_model_weights)
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

		# ===================================================================== #
		#                              Critic Model                             #
		# ===================================================================== #		

		self.critic_state_input, self.critic_action_input, \
			self.critic_model = self.create_critic_model()
		_, _, self.target_critic_model = self.create_critic_model()

		self.critic_grads = tf.gradients(self.critic_model.output, 
			self.critic_action_input) # where we calcaulte de/dC for feeding above
		
		# Initialize for later gradient calculations
		self.sess.run(tf.initialize_all_variables())

	# ========================================================================= #
	#                              Model Definitions                            #
	# ========================================================================= #

	def create_actor_model(self):
		state_input = Input(shape=self.env.observation_space.shape)
		h1 = Dense(512, activation='relu', name="act1")(state_input)
		h2 = Dense(512, activation='relu', name="act2")(h1)
		h3 = Dense(512, activation='relu', name="act3")(h2)
		h4 = Dense(512, activation='relu', name="act4")(h3)
		output = Dense(self.env.action_space.shape[0], activation='tanh')(h4)
		
		model = Model(inputs=state_input, outputs=output)
		adam  = Adam(lr=0.001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, model

	def create_critic_model(self):
		state_input = Input(shape=self.env.observation_space.shape, name="state_ip")
		state_h1 = Dense(512, activation='relu', name="crt1")(state_input)
		state_h2 = Dense(512, activation='relu', name="crt2")(state_h1)
		state_h3 = Dense(512)(state_h2)
		
		action_input = Input(shape=self.env.action_space.shape, name="action_ip")
		action_h1    = Dense(512, name="crt3")(action_input)
		
		merged    = Add()([state_h3, action_h1])
		merged_h1 = Dense(512, activation='relu', name="crt4")(merged)
		output = Dense(1, activation='linear', name="crt5")(merged_h1)
		model  = Model(inputs=[state_input,action_input], outputs=output)
		
		adam  = Adam(lr=0.001)
		model.compile(loss="mse", optimizer=adam)
		return state_input, action_input, model

	# ========================================================================= #
	#                               Model Training                              #
	# ========================================================================= #

	def remember(self, cur_state, action, reward, new_state, done):
		self.memory.append([cur_state, action, reward, new_state, done])

	def _train_actor(self, samples):
		for sample in samples:
			cur_state, action, reward, new_state, _ = sample
			predicted_action = self.actor_model.predict(cur_state)
			grads = self.sess.run(self.critic_grads, feed_dict={
				self.critic_state_input:  cur_state,
				self.critic_action_input: predicted_action
			})[0]

			self.sess.run(self.optimize, feed_dict={
				self.actor_state_input: cur_state,
				self.actor_critic_grad: grads
			})
            
	def _train_critic(self, samples):
		#print [r for x,y,r,z,c in samples]
		for sample in samples:
			cur_state, action, reward, new_state, done = sample
			if not done:
				target_action = self.target_actor_model.predict(new_state)
				future_reward = self.target_critic_model.predict(
					[new_state, target_action])[0][0]
				reward += self.gamma * future_reward
			#print list(cur_state[0]), action, reward
			# print "fitting reward:{}".format(reward)
			self.critic_model.fit({'state_ip':cur_state, 'action_ip':action}, np.array([reward]), verbose=0)
		
	def train(self):
		batch_size = 64
		if len(self.memory) < batch_size:
			return
		# print "Training"
		rewards = []
		samples = random.sample(self.memory, batch_size)
		self._train_critic(samples)
		self._train_actor(samples)

	# ========================================================================= #
	#                         Target Model Updating                             #
	# ========================================================================= #

	def _update_actor_target(self):
		actor_model_weights  = self.actor_model.get_weights()
		actor_target_weights = self.target_critic_model.get_weights()
		
		for i in range(len(actor_target_weights)):
			actor_target_weights[i] = actor_model_weights[i]
		self.target_critic_model.set_weights(actor_target_weights)

	def _update_critic_target(self):
		critic_model_weights  = self.critic_model.get_weights()
		critic_target_weights = self.critic_target_model.get_weights()
		
		for i in range(len(critic_target_weights)):
			critic_target_weights[i] = critic_model_weights[i]
		self.critic_target_model.set_weights(critic_target_weights)		

	def update_target(self):
		self._update_actor_target()
		self._update_critic_target()

	# ========================================================================= #
	#                              Model Predictions                            #
	# ========================================================================= #

	def act(self, cur_state):
		# print 'self.epsilon: {}'.format(self.epsilon)
		if np.random.random() < self.epsilon:
			# print 'trying to fetch random action'
			return self.env.randomAction()
		return self.actor_model.predict(cur_state)

	def load_trained_model(self, dir, epoch):
		print "Loading weights from: {}\nFor Epoch: {}".format(dir, epoch)
		self.actor_model.load_weights(dir+'/actor/w_E{}.h5'.format(epoch))
		self.target_actor_model.load_weights(dir+'/actor/w_E{}.h5'.format(epoch))
		self.critic_model.load_weights(dir+'/critic/w_E{}.h5'.format(epoch))
		self.target_critic_model.load_weights(dir+'/critic/w_E{}.h5'.format(epoch))
		print "Weights Loaded"



