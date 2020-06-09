import copy
import numpy as np

# 4 rooms
ROOM_1 = [[1,7], [2,7], [3,7], [4,7], [5,7],
            [1,8], [2,8], [3,8], [4,8], [5,8],
            [1,9], [2,9], [3,9], [4,9], [5,9],
            [1,10], [2,10], [3,10], [4,10], [5,10],
            [1,11], [2,11], [3,11], [4,11], [5,11]]

ROOM_2 = [[7,6], [8,6], [9,6], [10,6], [11,6],
            [7,7], [8,7], [9,7], [10,7], [11,7],
            [7,8], [8,8], [9,8], [10,8], [11,8],
            [7,9], [8,9], [9,9], [10,9], [11,9],
            [7,10], [8,10], [9,10], [10,10], [11,10],
            [7,11], [8,11], [9,11], [10,11], [11,11]]

ROOM_3 = [[7,1], [8,1], [9,1], [10,1], [11,1],
            [7,2], [8,2], [9,2], [10,2], [11,2],
            [7,3], [8,3], [9,3], [10,3], [11,3],
            [7,4], [8,4], [9,4], [10,4], [11,4]]

ROOM_4 = [[1,1], [2,1], [3,1], [4,1], [5,1],
            [1,2], [2,2], [3,2], [4,2], [5,2],
            [1,3], [2,3], [3,3], [4,3], [5,3],
            [1,4], [2,4], [3,4], [4,4], [5,4],
            [1,5], [2,5], [3,5], [4,5], [5,5]]
ROOMS = [ROOM_1, ROOM_2, ROOM_3, ROOM_4]

class HallwayOption:
    env = None

    def __init__(self, goal_type, room):
        if self.env is None:
            raise ValueError("Option environment not set")
        # two types of goals:
        # type 0 -> walls on left and right
        # type 1 -> walls on top and bottom
        self.goal_type = goal_type
        # set goal state
        self.goal = self.env.hallways[room-1][goal_type]
        self.create_init_set(goal_type, room)

    def create_init_set(self, goal_type, room):
        self.init_set = copy.deepcopy(ROOMS[room-1])
        if goal_type == 0:
            self.init_set.append(self.env.hallways[room-1][1])
        elif goal_type == 1:
            self.init_set.append(self.env.hallways[room-1][0])

    def policy(self, state):
        if state not in self.init_set:
            raise ValueError("State not in option's state set!")
            
        x, y = state
        if self.goal_type == 0:
            if x != self.goal[0]:
                if x > self.goal[0]:
                    action = 3
                elif x < self.goal[0]:
                    action = 2
            elif y != self.goal[1]:
                if y > self.goal[1]:
                    action = 1
                elif y < self.goal[1]:
                    action = 0
            
        elif self.goal_type == 1:
            if y != self.goal[1]:
                if y > self.goal[1]:
                    action = 1
                elif y < self.goal[1]:
                    action = 0

            elif x != self.goal[0]:
                if x > self.goal[0]:
                    action = 3
                elif x < self.goal[0]:
                    action = 2
        
        return action

    def run(self, state, algorithm="SMDP Learning", multi_step_options=None, Q_table=None, alpha=None):
        if algorithm == "SMDP Learning":
            return self.SMDP_Q_Learning(state)
        
        elif algorithm == "intra option learning":
            return self.intra_option_learning(state, multi_step_options, Q_table, alpha)

    def SMDP_Q_Learning(self, state):
        # accumulated reward while executing option
        R = 0
        # time steps till option terminates
        k = 0
        option_on = True
        while option_on:
            if state not in self.init_set:
                option_on = False
                break
            else:
                # choose action according to option policy
                action = self.policy(state)
                # take action in environment
                state, r, done, _ = self.env.step(action)
                # accumulate reward
                R += (self.env.gamma**k)*r
                # update time steps
                option_on = not done
                k += 1

        return state, R, done, k

    def intra_option_learning(self, state, multi_step_options, Q_table, alpha):
        R = 0
        k = 0
        option_on = True
        while option_on:
            if state not in self.init_set:
                option_on = False
                break
            else:
                # choose action according to option policy
                action = self.policy(state)
                # take action in environment
                new_state, r, done, _ = self.env.step(action)
                # get consistent options
                consistent_opts = find_consistent_options(state, action, multi_step_options)
                # valid options for new state
                valid_opts = valid_options(new_state, multi_step_options)
                # update for the consistent options
                x, y = state
                x_new, y_new = new_state
                for o in consistent_opts:
                    # option continues in next state i.e. β(sₜ₊₁) = 0
                    if new_state in multi_step_options[o-4].init_set:
                        Q_bar = Q_table[x_new, y_new, o]
                    # option terminates in next state i.e. β(sₜ₊₁) = 1
                    else:
                        Q_bar = np.max(Q_table[x_new, y_new, valid_opts])
                    
                    # update consistent option
                    Q_table[x, y, o] += alpha*(r + self.env.gamma*Q_bar - Q_table[x, y, o])
                
                state = new_state
                # accumulate reward
                R += (self.env.gamma**k)*r
                k += 1
                option_on = not done

        return state, R, done, k

                

def find_consistent_options(state, action, multi_step_options):
    # find consistent policies in s_t
    i = 4
    consistent_opts = []
    for option in multi_step_options:
        if state in option.init_set:
            if  option.policy(state) == action:
                consistent_opts.append(i)
        
        i += 1

    return consistent_opts

# given state, return valid options
def valid_options(state, multi_step_options):
    i = 0
    # primitive actions are always valid options
    valid_opts = [0, 1, 2, 3]
    for option in multi_step_options:
        # if state is in option's initiation set, option is valid
        if state in option.init_set:
            valid_opts.append(i+4)
        
        i += 1
    
    return valid_opts