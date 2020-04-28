import gym
from gym import error, spaces, utils
import numpy as np

class RoomWorld(gym.Env):
    def __init__(self):
        """
        Observation space is a (2,) tuple of discrete values that range in [0, 11],
        that describes the (x,y) position of the agent in the gridworld
        """
        self.observation_space = spaces.Tuple((spaces.Discrete(13), spaces.Discrete(13)))

        """
        Action space is scalar discrete with values in the range [0, 3], where:
        0 -> go up
        1 -> go down
        2 -> go right
        3 -> go left
        """
        self.action_space = spaces.Discrete(4)

        # parameters of the room world
        self.create_parameters()

        self.state = None
        self.goal = None
        self.initial_state_fix = False

    def create_parameters(self):
        # mark the outer walls
        self.walls = []
        for i in range(13):
            # top and bottom walls
            self.walls.append([i,0])
            self.walls.append([i,12])
            # left and right walls
            if i > 0 and i < 12:
                self.walls.append([0,i])
                self.walls.append([12,i])

        # inner walls
        self.walls.extend([[6,1], [6,3], [6,4], [6,5], 
        [6,6], [6,7], [6,8], [6,10], [6,11], [7,5], 
        [8,5], [10,5], [11,5], [5,6], [4,6], [3,6], 
        [1,6]])

        # hallways (arranged according to type, room wise)
        self.hallways = [[[2,6], [6,9]], [[9,5], [6,9]], [[9,5], [6,2]], [[2,6], [6,2]]]

        # goal states
        self.G2 = [9,3]
        self.G1 = [9,5]

        # discount factor
        self.gamma = 0.9

    def reset(self):
        if self.initial_state_fix:
            self.state = (3,3)
            
        else:
            init  = False
            while not init: 
                x0, y0 = self.observation_space.sample()
                if [x0, y0] not in self.walls:
                    self.state = (x0, y0)
                    init = True

        return list(self.state)

    def step(self, action):
        # check if goal state is set
        if self.goal is None:
            raise ValueError("Goal state not set")
        
        done = False

        self.transition(action)

        r, done = self.reward()

        return list(self.state), r, done, {}

    def reward(self):
        r = 0
        done = False

        # reward for reaching goal state
        if list(self.state) == self.goal:
            r = 1
            done = True
        
        return r, done

    def transition(self, action):
        x, y = self.state

        # random action selection
        rand_prob = 1/9
        action_prob = rand_prob*np.ones((4,))
        action_prob[action] = 2/3
        action = np.random.choice(np.arange(4), 1, p=action_prob)

        # transition (with clipping if at walls)
        if action == 0:
            action_2d = [0,1]
        elif action == 1:
            action_2d = [0,-1]
        elif action == 2:
            action_2d = [1,0]
        elif action == 3:
            action_2d = [-1,0]
        action_2d = np.array(action_2d)

        if [x+action_2d[0], y+action_2d[1]] not in self.walls:
            x += action_2d[0]
            y += action_2d[1]

        self.state = (x, y) 