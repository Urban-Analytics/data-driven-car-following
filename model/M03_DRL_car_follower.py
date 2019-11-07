import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

ENV_NAME = 'Pendulum-v0'

env = gym.make(ENV_NAME)

