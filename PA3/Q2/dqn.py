import gym
import torch.nn as nn
import numpy as np
import random
import copy

class DQN:
    REPLAY_MEMORY_SIZE = 10000
    BATCH_SIZE = 10
    EPISODES = 2000

    def __init__(self, env, model):
        self.model = model
        self.env = gym.make(env)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n

        # init replay buffer
        self.replay_buffer = ReplayMemory(self.REPLAY_MEMORY_SIZE, self.BATCH_SIZE)
        
        # init target network
        self.target_network = copy.deepcopy(model)

    # calculate Q values and the approximate targets from experience
    def loss_function_data(self):
        experiences = self.replay_buffer.sample()
        # format experiences for easier handling
        data = np.zeros((self.BATCH_SIZE, 5), dtype='object')
        for i in range(self.BATCH_SIZE):
            data[i,:] = experiences[i]

        # calculate approximate targets
        y = np.zeros((self.BATCH_SIZE,))
        for i in range(self.BATCH_SIZE):
            y[i] = 

        


# to handle experiences
class ReplayMemory:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.pos = 0

    def add(self, s, a, r, new_s, done):
        e = [s, a, r, new_s, int(done)]
        # if replay buffer not full, append new experience
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(e)
        else:
            self.buffer[self.pos] = e
        self.pos += 1
        self.pos = self.pos % self.buffer_size

    def sample(self):
        return random.sample(self.buffer, self.batch_size)