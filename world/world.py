import numpy as np
import time

class World:
	def __init__(render=False):
		# States represented by (x, y, theta:in degrees)
		self.initialState = (0, 0, 0)
		self.destination = (100, 100, 100) 
		self.t_prev, self.t_next = None, None
		return True

	def setDestination(state):
		self.destination = state

	def initializeState(state):
		self.initialState = state
		self.currentState = self.initialState

	def reward(state):
		if self.currentState == self.destination:
			return 100, True
		distance = self.calcDistance()
		return 100 - distance, False

	def calculateTotalDistance():
		self.totalDistance = calcDistance(init=True)

	def calcDistance(init=False):
		if init:
			self.calculateTotalDistance()
		dstate = list(self.destination) - list(self.currentState)
		dx, dy = dstate[0], dstate[1]
		dtheta = (dstate[2] - currentState[2] + 180) % 180 - 180
		return np.linalg.norm([dx, dy, dtheta])

	def reset():
		self.currentState = self.initialState
		self.totalDistance = None
		return self.initialState

	def initTime():
		if self.t_next == None:
			self.t_next = time.time()
			if self.t_prev == None:
				self.t_prev = time.time()

	def step(state, action):
		v, w = action
		th = self.currentState[2]*np.pi/180
		x = self.currentState[0]
		y = self.currentState[1]
		self.initTime()
		dt = self.t_next - self.t_prev
		reward, done = self.reward(state)
		d_state = [v*np.cos(th), v*sin(th), w]
		next_state = list(state) + d_state
		next_state = tuple(next_state)
		self.t_prev = self.t_next
		self.t_next = None
		return next_state, reward, done
