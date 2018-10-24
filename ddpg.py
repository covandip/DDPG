# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:54:33 2018

@author: Cameron
"""

from replaybuffer import ReplayBuffer
from actor import Actor
from critic import Critic
import tensorflow
from keras import backend as k_backend


class DDPG:
    def __init__(self, env, batch_size, mem_size, 
            discount, actor_params, critic_params):
       self._batch_size = batch_size
       self._mem_size = mem_size
       self._discount = discount
       self._sess = tensorflow.Session()
       k_backend.set_session(self._sess)
       self._env = env
       self._state_dim = env.observation_space.shape[0]
       self._action_dim = env.action_space.shape[0]
       self._action_min = env.action_space.low
       self._action_max = env.action_space.high
       self._state_min = env.observation_space.low
       self._state_max = env.observation_space.high
       self._actor = Actor(self._sess, self._state_dim, self._action_dim, self._action_min, 
                           self._action_max, actor_params)
       self._critic = Critic(self._sess, 0.5, self._state_dim, self._action_dim, critic_params)
       self._memory = ReplayBuffer(mem_size)

    def get_action(self, state):
        return self._actor._model.predict(state)

    def train(self):
        '''
        No training takes place until the replay buffer contains
        at least batch size number of experiences
        '''

        if(self._memory.size() > self._batch_size):
            self._train()
    
    def _train(self):
        states, actions, rewards, done, next_states = self._memory.sample(self._batch_size)
        self._train_critic(states, actions, rewards, done, next_states)
        action_gradients = self._critic.action_gradients(states, actions)
        self._actor.train(states, action_gradients)
        
    def q_estimate(self, state, action):
        return self._critic._model.predict(state, action)
    
    def _get_q_targets(self, next_states, done, rewards):
        '''
        q = r if done else =  r + gamma * qnext
        '''
        # use actor network to determine the next action under current policy
        # estimate Q values from the critic network

        actions = self.get_action(next_states)
        qnext = self.q_estimate(next_states, actions)

        q_targets = [reward if end else reward * self._discount * next_q
                for (reward, next_q, end)
                in zip(rewards, qnext, done)]
        return q_targets

    def _train_critic(self, states, actions, rewards, done, next_states):
        q_targets = self._get_q_targets(next_states, done, rewards)
        self._critic.train(states, actions, q_targets)
    
    def experience(self, state, action, reward, done, next_state):
        # store in replay buffer
        self._memory.add(state, action, reward, done, next_state)

        self.train()
