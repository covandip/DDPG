import tensorflow as tf
import gym
from ddpg import DDPG as Agent
env = gym.make("MountainCarContinuous-v0")
env.reset()
keel = Agent(env,100,10000,0.99,(100,100,0.9),(100,100,0.9))


