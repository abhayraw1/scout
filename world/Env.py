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
		self.epsLen	= 10 # in cm
		self.epsyaw = 5.*np.pi/180.
		self.dt 	= 1
		self.ticks	= 0
		self.tPnlty	= .001
		self.rewardParameter = 200
		self.action_space = np.zeros((2, ))
		self.observation_space = np.zeros((64, ))
		self.maxV	= .4 # in m/s
		self.maxW	= .5 # in rad/s
		self.death	= 'Time'
		self.info	= ''
		self.ERASE_LINE = '\x1b[2K'

	def reset(self):
		'''	Resets the environment state.
			Sets a new goal point and a 
			new spawn point for the agent.'''
		self.prevVW = {'v': 0, 'w': 0}
		self.halt()
		resetPose	= jderobot.Pose3DData()
		resetPose.x		= -3.5
		resetPose.y		= -0.5
		resetPose.z 	= 0
		resetPose.yaw 	= 0
		resetPose.roll 	= 0
		resetPose.pitch = 0
		self.pose3d.pose3d.proxy.setPose3DData(resetPose)
		self.goal 	= goalstates.getRandomGoal(56)
		print "GOAL: {}".format(self.goal)
		self.state	= self.getCurrentState()
		# #print self.goal, self.state
		return np.array(self.state)

	def getCurrentState(self):
		laserData 	= self.laser.getLaserData().distanceData[:180:3]
		laserData	= [x/2000. for x in laserData] # divide by max possible value.
		pose 		= np.array([self.pose3d.getPose3d().x, self.pose3d.getPose3d().y,\
								self.pose3d.getPose3d().yaw])
		# pose 		= [pose[k] for k in ('x','y','yaw')]
		goal 		= np.array([self.goal[k] for k in ('x','y','yaw')])
		distance 	= np.linalg.norm(goal[:2]-pose[:2])
		theta 		= np.arctan2((goal[1]-pose[1]),(goal[0]- pose[0]))
		vel 		= [self.prevVW[k] for k in ('v','w')]
		self.info	= 'Distance: {}\t'.format(distance)
		# print 'Distance: {}\t Theta:{}'.format(distance, theta)
		return laserData + [distance, theta] + vel

	def step(self, action):
		prevState		= self.state[:]
		self.act(action)
		self.ticks		+= 1
		reward, done 	= self.reward(prevState)
		self.info 		+= 'Reward: {}'.format(reward)
		sys.stdout.write(self.ERASE_LINE)
		sys.stdout.write('\r'+self.info)
		sys.stdout.flush()
		self.info 		= ''
		return np.array(self.state), reward, done

	def act(self, action):
		#print action
		v, w = action[0]
		# because of sigmoid action the values of v/w are in [0,1]
		# we need to make w to be in range [-1,1]
		v = (v + 1)*.5    
		self.motors.sendV(v*self.maxV + 1e-5)
		self.motors.sendW(w*self.maxW)
		# print "sending v: {}\t w:{}".format(v*self.maxV + 1e-5, w*self.maxW)
		# print "Velocity"
		self.prevVW['v'] = v
		self.prevVW['w'] = w
	
	def reward(self, prevState):
		# goal pose
		gVector = np.array([self.goal['x'], self.goal['y']])
		# pose at time step t-1
		pVector	= np.array([prevState[60], prevState[61]])
		# pose at time step t
		self.state 		= self.getCurrentState()
		cVector = np.array([self.state[60], self.state[61]])
		dT2		= np.linalg.norm(gVector - pVector)*100 # in cm
		dT1		= np.linalg.norm(gVector - cVector)*100 # in cm 
		self.info += 'c_pose: {}\tp_pose: {}'.format(cVector, pVector)
		# #print dT2, dT1
		done = False
		if self.hasReached(dT1) and self.hasStopped():
			self.death = "Reached"
			print "Goal Reached!!"
			reward = 500
			done = True
		elif self.isDead():
			self.death = "Dead"
			self.halt()
			reward = -5000
			done = True
		else:
			reward 	= (dT2 - dT1) * self.rewardParameter
			reward 	= reward
		# print "Current Reward: {}\tDone: {}".format(reward, done)
		return reward, done
		pass

	def isDead(self):
		'''	returns true if any of the laser 
			values is less than a certain 
			threshold(in mm). '''
		# print "Dead."
		lazerData 	= self.laser.getLaserData().distanceData[:180]
		# print np.min(lazerData)
		# print lazerData
		return np.min(lazerData) <= 180 # 18cm from center of robot

	def hasStopped(self):
		return self.prevVW['v'] == 0 and self.prevVW['w'] == 0

	def hasReached(self, distance):
		return distance <= self.epsLen# and np.abs(diffAngle) < epsyaw

	def halt(self):
		self.motors.sendV(0)
		self.motors.sendW(0)

	def randomAction(self):
		v = np.random.random() * self.maxV
		w = (2*np.random.random()-1) * self.maxW
		# print "randomAction [{}, {}]".format(v,w)
		return np.array([v, w])
