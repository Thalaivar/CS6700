import gym
from gym import spaces, error, utils
import numpy as np

class VishamC(gym.Env):
    def __init__(self):
        # observation space limits
        self.state_bound = 1
        bounds = self.state_bound*np.ones((2,))
        self.observation_space = spaces.Box(low=-bounds, high=bounds)

        # action space dims
        self.action_bound = 0.025
        bounds = self.action_bound*np.ones((2,))
        self.action_space = spaces.Box(low=-bounds, high=bounds, dtype=np.float64)

        self.tol = 1e-4
        
        self.state = None

    def reset(self):
        x, y = -1 + 2*np.random.rand(2)
        self.state = (x, y)

        return np.array(self.state)

    def step(self, action):
        state = self.state
        x, y = state

        # get reward
        gamma = 10
        r = 0.5*x**2 + gamma*0.5*y**2

        x += action[0]
        y += action[1]
        done = False
        if np.sqrt(x**2 + y**2) < self.tol:
            done = True

        self.state = (x, y)

        return np.array(self.state), r, done, {}
