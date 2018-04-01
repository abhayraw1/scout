import sys
import comm
import config
import jderobot
import goalstates
import numpy as np

class Gazeboworld:

	@staticmethod
	def angDiff(an1, an2):
		dTh	= an1 - an2
		if dTh > np.pi:
			dTh += -2*np.pi
		elif dTh < -np.pi:
			dTh += 2*np.pi
		else:
			pass
		return dTh

	def __init__(self):
		cfg 		= config.load('/home/abhay/Desktop/autopark.cfg')
		jdrc		= comm.init(cfg, 'Autopark')
		self.motors = jdrc.getMotorsClient ("Autopark.Motors")
		self.pose3d = jdrc.getPose3dClient("Autopark.Pose3D")
		self.laser 	= jdrc.getLaserClient("Autopark.Laser").hasproxy()
		self.goal 	= {}
		self.state 	= {}
		self.epsLen	= 10
		self.epsyaw = 5.*np.pi/180.
		self.dt 	= 1
		self.ticks	= 0
		self.tPnlty	= .001
		self.rewardParameter = 1
		self.action_space = np.zeros((2, ))
		self.observation_space = np.zeros((68, ))
		self.maxV	= 5
		self.maxW	= 5

	def reset(self):
		'''	Resets the environment state.
			Sets a new goal point and a 
			new spawn point for the agent.'''
		self.prevVW = {'v': 0, 'w': 0}
		self.halt()
		resetPose	= jderobot.Pose3DData()
		resetPose.x	= -3.5
		resetPose.y	= -0.5
		self.pose3d.pose3d.proxy.setPose3DData(resetPose)
		self.goal 	= goalstates.getRandomGoal()
		self.state	= self.getCurrentState()
		print self.goal, self.state
		return np.array(self.state)

	def getCurrentState(self):
		laserData 	= self.laser.getLaserData().distanceData[:360:6]
		laserData	= [x/2000. for x in laserData] # divide by max possible value.
		pose 		= [self.pose3d.getPose3d().x, self.pose3d.getPose3d().y, self.pose3d.getPose3d().yaw]
		# pose 		= [pose[k] for k in ('x','y','yaw')]
		goal 		= [self.goal[k] for k in ('x','y','yaw')]
		vel 		= [self.prevVW[k] for k in ('v','w')]
		return laserData + pose + goal + vel

	def step(self, action):
		prevState		= self.state
		self.act(action)
		self.ticks		+= 1
		self.state 		= self.getCurrentState()
		reward, done 	= self.reward(prevState)
		return np.array(self.state), reward, done

	def isDead(self):
		'''	returns true if any of the laser 
			values is less than a certain 
			threshold(in mm). '''
		lazerData 	= self.laser.getLaserData().distanceData[:360]
		return np.min(lazerData) <= 180 # 18cm from center of robot

	def hasReached(self, distance):
		return distance <= self.epsLen# and np.abs(diffAngle) < epsyaw

	def act(self, action):
		print action    
		v, w = action[0]
		self.prevVW['v'] = v
		self.prevVW['w'] = w
		self.motors.sendV(v+1e-5)
		self.motors.sendW(w+1e-5)
		pass

	def halt(self):
		self.motors.sendV(0)
		self.motors.sendW(0)

	def reward(self, prevState):
		# goal pose
		gVector = np.array([self.goal['x'], self.goal['y']])
		# pose at time step t-1
		pVector	= np.array([prevState[60], prevState[62]])
		# pose at time step t
		cVector = np.array([self.state[60], self.state[62]])
		dT2		= np.linalg.norm(gVector - pVector)*100
		dT1		= np.linalg.norm(gVector - cVector)*100
		print dT2, dT1
		if self.hasReached(dT1):
			return 50, True
		elif self.isDead():
			return -50, True
		else:
			reward 	= (dT2 - dT1) * self.rewardParameter
			return reward, False
		pass

	def randomAction(self):
		v = np.random.random() * self.maxV
		w = np.random.random() * self.maxW
		return np.array([v, w])