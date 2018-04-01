from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from world import World
from collections import deque
import numpy as np
import random
from agent import Agent

class Pibot(Agent):
	def __init__(self, state_size):
		self.state_size = state_size
		self.action_size = 
		Agent.__init__(self, self.state_size, self.action_size)