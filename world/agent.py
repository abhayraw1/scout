import pickle
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from world import World
from collections import deque
import numpy as np
import random

class Agent:
	def __init__(self, state_size, action_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=4*50000)
		self.gamma = 0.95    # discount rate
		self.epsilon = 1.0  # exploration rate
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.9 
		self.learning_rate = 0.001
		self.model = self._build_model()
	
	def _build_model(self):
		model = Sequential()
		model.add(Dense(24, input_dim=self.state_size, activation='relu'))
		model.add(Dense(48, activation='relu'))
		model.add(Dense(48, activation='relu'))
		model.add(Dense(self.action_size, activation='linear'))
		model.compile(loss='mse',
					  optimizer=Adam(lr=self.learning_rate))
		return model
	
	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))
	
	def act(self, state, explore=True):
		if np.random.rand() <= self.epsilon and explore:
			return random.randrange(8), random.randrange(8, 72)
		act_values = self.model.predict(state)[0]
		v, w = act_values[:8], act_values[8:]
		return np.argmax(v), np.argmax(w)  # returns action
	
	def max_q(self, next_state):
		prediction = self.model.predict(next_state)[0]
		p_v, p_w = prediction[:8], prediction[8:]
		return np.argmax(p_v[0]), np.argmax(p_w[1])

	def replay(self, batch_size):
		print 'in_replay'
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state, done in minibatch:
			target = list(reward)
			if not done:
				r = self.max_q(next_state)
				target[0] = reward[0] + self.gamma * r[0]
				target[1] = reward[1] + self.gamma * r[1]
			target_f = self.model.predict(state)
			target_f[0][action[0]] = target[0]
			target_f[0][action[1]] = target[1]
			self.model.fit(state, target_f, epochs=1, verbose=0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def saveWeights(self, filename):
		weights = agent.model.save_weights('my_model_weights.h5')
		print 'Weights Saved'
		# f = open(filename, 'w+')
		# f.write(json.dumps(weigths))
		# f.close()

	def setWeights(self, filename):
		self.model.load_weights(filename)


if __name__ == "__main__":
	episodes = 1000
	env = World(20, 20)
	agent = Agent(3, 72)
	for e in range(episodes):
		min_dist = None
		min_th = None
		state = env.reset()
		state = np.reshape(state, [1, 3])
		print 'Episode: {}'.format(e)
		for time_t in range(50000):
			action = agent.act(state)
			next_state, reward, done = env.step(action)
			next_state = np.reshape(next_state, [1, 3])
			try:
				if reward[0] > 0 or True:
					agent.remember(state,  action, reward, next_state, done)
			except:
				print reward
				pass 
			state = next_state
			if type(min_dist) == type(None) or min_dist > env.dth:
				min_dist = env.d
			if type(min_th) == type(None) or min_th > env.dth:
				min_th = env.dth
			# env.log()
			if done:
				print("episode: {}/{}, score: {}"
					  .format(e, episodes, time_t))
				break
		print 'Episode: {}\tMin dist:{}\tMin theta:{} '.format(e, min_dist, min_th) 
		agent.replay(256)
	agent.saveWeights('weights.txt')
	print '\n'

	# f = open('tuples.txt', 'w+')
	# data = pickle.dumps(agent.memory)
	# f.write(data)
	# f.close()