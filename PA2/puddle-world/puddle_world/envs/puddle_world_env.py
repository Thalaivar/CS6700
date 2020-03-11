import gym
from gym import error, spaces, utils
import numpy as np

class PuddleWorld(gym.Env):
    def __init__(self):
        """
        Observation space is a (2,) tuple of discrete values that range in [0, 11],
        that describes the (x,y) position of the agent in the gridworld
        """
        OBS_DIM = 12
        self.observation_space = spaces.Tuple(spaces.Discrete(12), spaces.Discrete(12))
        
        """
        Action space is scalar discrete with values in the range [0, 3], where:
        0 -> go up
        1 -> go down
        2 -> go right
        3 -> go left
        """
        ACTION_DIM = 4
        self.action_space = spaces.Discrete(ACTION_DIM)

        # parameters of the puddle world
        # wind on by default
        self.wind = True
        # dense distance reward off by default
        self.distance_reward = False
        # goal state for the task
        self.goal_state = None

        # puddle coordinates
        self.DEPTH_1 = [(3,3), (4,3), (5,3), (6,3), (7,3), (7,4), (7,5), (8,5), (8,6),
                        (8,7), (8,8), (8,9), (7,9), (6,9), (5,9), (4,9), (3,9), (3,8),
                        (3,7), (3,6), (3,5), (3,4)]
        self.DEPTH_2 = [(4,4), (5,4), (6,4), (6,5), (6,6), (7,6), (7,7), (7,8), (6,8),
                        (5,8), (4,8), (4,7), (4,6), (4,5)]
        self.DEPTH_3 = [(5,5), (5,6), (5,7), (6,7)]

        self.state = None

    def reset(self):
        initial_states = [(0,0), (0,1), (0,5), (0,6)]
        initial_choice = np.random.choice(np.arange(4), 1)
        self.state = initial_states[initial_choice]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        state = self.state
        x, y = puddle_world_transition(state, action, self.wind)

    def reward(self):
        # reward for reaching goal state
        if self.state == self.goal_state:
            return 10
        
        # penalize entering the puddle
        if self.state in self.DEPTH_1:
            return -1
        elif self.state in self.DEPTH_2:
            return -2
        elif self.state in self.DEPTH_3:
            return -3
        


    
def puddle_world_transition(state, action, wind):
    x, y = state

    # random action selection
    rand_prob = 0.1/3
    action_prob = rand_prob*np.ones((4,))
    action_prob[action] = 0.9
    action = np.random.choice(np.arange(4), 1, p=action_prob)

    # transitions (with clipping if at edge of world)
    if action == 0:
        # go up
        if y < 11:
            y += 1
    elif action == 1:
        # go down
        if y > 0:
            y -= 1
    elif action == 2:
        # go right
        if x < 11:
            x += 1
    elif action == 3:
        # go left
        if x > 0:
            x -= 1
    
    # random right transition
    if wind:
        if np.random.rand() > 0.5:
            if x < 11:
                x += 1

    return x, y