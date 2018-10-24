import tensorflow as tf
import gym
from ddpg import DDPG as Agent

env = gym.make("MountainCarContinuous-v0")
keel = Agent(env,100,10000,0.99,(100,100,0.9),(100,100,0.9))

while(True):
    starting_state = env.reset()
    action = keel.get_action(starting_state)
    next_state, reward, done = env.observation(action)
    keel.experience(starting_state, action, reward, done, next_state)
    keel.train()
