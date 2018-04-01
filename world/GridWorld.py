import numpy as np

class GridWorld:
	def __init__(self, rows, cols, destination=None, initialState=None):
		self.rows 		= rows
		self.cols 		= cols
		self.agentPos	= np.array([0,0])
		self.setInitState(initialState)
		self.setdestination(destination)
		self.populateActions()
		print self.action

	def populateActions(self):
		shift = (-1, 0, 1)
		self.action = [(i,j) for i in shift for j in shift]

	def move(self, actionIdx):
		self.currentActionId = actionIdx
		self.agentPos = self.agentPos + self.action[actionIdx]
		self.calculateReward()

	def setInitState(self, initialState):
		if isinstance (initialState, type(None)):
			self.state = np.zeros((self.rows, self.cols))
		elif state.shape == (self.rows, self.cols):
			self.state = np.array(initialState)
		else:
			print "There's some error in the initial state provided."

	def setdestination(self, destination):
		if isinstance (destination, type(None)):
			self.destination = self.rows-1, self.cols-1
		if np.any(np.array(destination) > self.state.shape):
			print "There's some error in the destination provided."
		else:
			self.destination = destination

	def calculateReward(self):
		self.reward		= 0
		if np.all(self.agentPos == self.destination):
			return 2
		if np.any(self.agentPos > self.state.shape):
			self.agentPos = self.agentPos - self.action[self.currentActionId]
			self.reward = -1
		
	def step(self, actionIdx):
		self.move(actionIdx)
		self.
		pass


a = GridWorld(4,4)
