import matplotlib.pyplot as plt
import numpy as np
from agent import Agent
from world import World
# import sys

def move(env, agent):
	state = env.currentState
	action = agent.act(np.reshape(state, [1,3]))
	distance, dth = env.calcDistance()
	print 'Action: {}\tDistance:{}'.format(action, distance)
	c, r, done = env.step(action)
	return done

def savePoints(points, filename):
	np.asarray(points)
	np.savetxt(filename, points)

if __name__ == "__main__":
	agent = Agent(3, 72)
	agent.model.load_weights('my_model_weights.h5')
	states = []
	start = [0,0,0]
	current = start[:]
	env = World(500, 500)
	env.setInitialState(World._randomPoint(500, 500))
	env.setDestination(World._randomPoint(500, 500))
	print 'Starting Point: {}\nFinal Point: {}'.format(env.initialPosition, env.destination)
	points = []
	while not move(env, agent):
		points.append(env.currentPosition)
	savePoints(points, 'points.csv')




	