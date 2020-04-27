import gym
import numpy as np
from options import HallwayOption

def main(goal_state):
    env = gym.make("room_world:room-v0")

    if goal_state == 1:
        env.goal = env.G1
    elif goal_state == 2:
        env.goal = env.G2

def train(env):
    # multi-step options
    multi_step_options = create_options(env)

    


# composite step function to include multi-step options
def step(env, option, state, multi_step_options):
    # for multi-step option
    if option > 3:
        # retrieve multi-step option
        option = multi_step_options[option-4]
        state, r, done = option.run(state)
    else:
        # primitive action
        state, r, done, _ = env.step(option)
    
    return state, r, done


# policy to choose a valid option given a state
def policy(state, options, Q_table):
    x, y = state
    valid_opts = valid_options(state, options)

    # epsilon - greedy option selection
    eps = 0.1
    if np.random.rand() < eps:
        # choose randomly from valid options
        option = np.random.choice(valid_opts, size = 1)
    else:
        # choose greedily from valid options
        max_option = valid_opts[0]
        max_Q = Q_table[x, y, valid_opts[0]]
        for o in valid_opts:
            if Q_table[x, y, o] > max_Q:
                max_Q = Q_table[x, y, o]
                max_option = o

        option = max_option

    return option

# given state, return valid options
def valid_options(state, options):
    i = 0
    # primitive actions are always valid options
    valid_opts = [0, 1, 2, 3]
    for option in options:
        # if state is in option's initiation set, option is valid
        if state in option.init_set:
            valid_opts.append(i+4)
        
        i += 1
    
    return valid_opts

# create all 8 multi-step options
def create_options(env):
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

    option_1_1 = HallwayOption(ROOM_1, env.hallways[2], 0, env, env.hallways[1])
    option_1_2 = HallwayOption(ROOM_1, env.hallways[1], 1, env, env.hallways[2])
    option_2_1 = HallwayOption(ROOM_2, env.hallways[2], 0, env, env.hallways[3])
    option_2_2 = HallwayOption(ROOM_2, env.hallways[3], 1, env, env.hallways[2])
    option_3_1 = HallwayOption(ROOM_3, env.hallways[3], 1, env, env.hallways[0])
    option_3_2 = HallwayOption(ROOM_3, env.hallways[0], 0, env, env.hallways[3])
    option_4_1 = HallwayOption(ROOM_4, env.hallways[0], 0, env, env.hallways[1])
    option_4_2 = HallwayOption(ROOM_4, env.hallways[1], 1, env, env.hallways[0])

    return [option_1_1, option_1_2, option_2_1, option_2_2, option_3_1, option_3_2,
            option_4_1, option_4_2]