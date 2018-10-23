# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:54:33 2018

@author: Cameron
"""

from replaybuffer import ReplayBuffer
from actor import Actor
from critic import Critic
import tensorflow
import gym

env = gym.make("MountainCarContinuous-v0")
env.reset()

class DDPG:
    def __init__(self, batch_size, mem_size, 
            discount, actor_params, critic_params):
       self._batch_size = batch_size
       self._mem_size = mem_size
       self._discount = discount
       self._sess = tensorflow.Session()

       self._actor = Actor(sess, actor_params)
       self._critic = Critic(sess, critic_params)
       self._memory = ReplayBuffer(mem_size)

    def get_action(self, state):
        return self._actor._model.predict(state)

