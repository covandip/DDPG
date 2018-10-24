# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 12:49:47 2018

@author: Cameron
"""

from collections import deque
import random
import numpy as np

class ReplayBuffer:
    
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
    
    def add(self, state,action, reward, done, next_state):
        if (len(self.buffer) < self.buffer_size):
            self.buffer.append((state,action,reward,done,next_state))
        else:
            self.buffer.popleft()
            self.buffer.append((state,action,reward,done,next_state))
    
    def sample(self, batch_size):
        """
        Returns a sample from the replay buffer
        
        :param batch_size: Int denoting how many experiences to sample
        """
    
        batch = []
        
        if (len(self.buffer) < batch_size):
            batch = random.sample(self.buffer, len(self.buffer))
        
        else:
            batch = random.sample(self.buffer, batch_size)
        # each experience is of the format (state, action, reward, time, next_state)

        state_batch = np.array([experience[0] for experience in batch])
        action_batch = np.array([experience[1] for experience in batch])
        reward_batch = np.array([experience[2] for experience in batch])
        done_batch  = np.array([experience[3] for experience in batch])
        next_state_batch = np.array([experience[4] for experience in batch])
        
        return state_batch, action_batch, reward_batch, done_batch, next_state_batch
    
    def flush(self):
        self.buffer.clear()
        
    def size(self):
        return len(self.buffer)
    
