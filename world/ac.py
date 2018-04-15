import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam

import tensorflow as tf
import time
import random
from collections import deque
from Env import Gazeboworld

class ActorCritic:
    def __init__(self, env):
        self.env            = env
        self.learning_rate  = 0.001
        self.epsilon        = 1.0
        self.epsilon_decay  = .9999
        self.gamma          = .95
        self.memory         = deque(maxlen=7500)
        self.target_actor_model   = self.create_actor_model()
        self.actor_model    = self.create_actor_model()
        self.target_critic_model  = self.create_critic_model()
        self.critic_model   = self.create_critic_model()

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
        return model

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
        return model

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])
     
    def _train_critic(self, samples):
        # print "Training Critic"
        states, actions, rewards = [[],[],[]]
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            states.append(list(cur_state[0]))
            actions.append(list(action[0]))
            rewards.append(reward)
        self.critic_model.fit({'state_ip':np.array(states), 'action_ip':np.array(actions)}, 
                np.array(rewards), epochs=10, verbose=0)
        
    def _train_actor(self, samples):
        # print "training actor---->"
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            target_action = self.target_actor_model.predict(new_state)
            q_t2 = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
            q_t1 = self.target_critic_model.predict(
                    [cur_state, action])[0][0]
            delta = reward + self.gamma*q_t2 - q_t1
            if delta > 0:
                self.actor_model.fit(cur_state, action, verbose=0)
        
    def train(self):
        batch_size = 64
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)
        # PRINT WEIGHTS OF SOME MODEL :
        # for layer in self.target_actor_model.layers:
        #     print layer.get_weights()
        self.update_target()


    def _update_actor_target(self):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_actor_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()
        
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.target_critic_model.set_weights(critic_target_weights)     

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

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
