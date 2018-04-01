import numpy as np
import time

class World:
		
	@staticmethod
	def _randomPoint(length, width):
		point = np.random.random(3)
		point = point * [width, length, 2*np.pi]
		point = map(int, point)
		return point

	@staticmethod
	def _project(a, b):
		return a[0]*b[0] + a[1]*b[1]

	def __init__(self, width=500, length=500, render=False):
		# States represented by (x, y, theta:in rad)
		self.width 				= width
		self.length				= length
		self.vision				= 20
		# self.initialState = World._randomState(length, width)
		# self.destination = World._randomState(length, width)

		self.destination		= np.array([500, 450, 172*np.pi/180]) 
		self.setInitialState(np.array([0,0,0]))
		# self.currentState

	def setDestination(self, point):
		self.destination = np.array(point)

	def setInitialState(self, point):
		self.initialPosition = np.array(point)
		self.currentPosition = np.array(point)
		self.currentState = self.getState() 

	def getState(self):
		r = self.destination[:2] - self.currentPosition[:2]
		dtheta = self.destination[2] - self.currentPosition[2]
		if dtheta > np.pi:
			dtheta = dtheta - 2*np.pi
		if dtheta < -np.pi:
			dtheta = dtheta + 2*np.pi
		if r[0] > self.vision:
			r[0] = self.vision
		if r[0] < -self.vision:
			r[0] = -self.vision
		if r[1] > self.vision:
			r[1] = self.vision
		if r[1] < -self.vision:
			r[1] = -self.vision
		return list(np.round(r)) + [dtheta]

	def updateState(self):
		self.currentPosition = self.nextPosition
		self.currentState = self.getState()

	def reward(self, dPosition):
		reward = World._project(dPosition, self.currentState)
		distance, dtheta = self.calcDistance()
		if distance < .5 and dtheta < .01:
			return 100, True
		self.d = distance
		self.dth = dtheta
		# print distance, dtheta
		return (reward, -np.abs(dtheta)), False

	def calcDistance(self, init=False):
		dx, dy, dth = np.array(self.destination) - np.array(self.currentPosition)
		return np.linalg.norm([dx, dy]), dth

	def reset(self):
		self.__init__()
		return self.currentState

	def step(self, action):
		v, w = action
		# print "V: {}\tW: {}".format(v, w)
		x, y, theta = self.currentPosition
		th = (w-8)*2.*np.pi/64 - np.pi
		if th > np.pi:
			th = th - 2*np.pi
		if th < -np.pi:
			th = th + 2*np.pi
		self.dPosition = [v*np.cos(theta), v*np.sin(theta), th]
		self.nextPosition = self.currentPosition + self.dPosition
		if self.nextPosition[2] > np.pi:
			self.nextPosition[2] = self.nextPosition[2] - 2*np.pi
		if self.nextPosition[2] < -np.pi:
			self.nextPosition[2] = self.nextPosition[2] + 2*np.pi
		self.updateState()
		self.re, done = self.reward(self.dPosition)
		return self.currentState, self.re, done

	def log(self):
		log 	= 	'Destination:\t\t{}\n' \
					'Current Position:\t{}\n' \
					'D Position:\t\t{}\n' \
					'Current State:\t\t{}\n' \
					'Reward:\t\t\t{}\r'
		log = log.format(self.destination, self.currentPosition, self.dPosition, self.currentState, self.re)
		print log
