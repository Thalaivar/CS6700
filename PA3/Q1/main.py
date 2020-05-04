import gym
import numpy as np
from options import HallwayOption

N_EPISODES = 1000
N_RUNS = 100

def main(goal_state, initial_state_fix, intra_option_learning):
    env = gym.make("room_world:room-v0")

    if goal_state == 1:
        env.goal = env.G1
    elif goal_state == 2:
        env.goal = env.G2

    env.initial_state_fix = initial_state_fix

    steps, Q_table, _ = train(env, intra_option_learning)

    np.save("results/results_Qtable_IO_G2", Q_table)
    np.save("results/results_steps_IO_G2", steps)
    
def train(env, intra_option_learning):
    alpha = 1/8

    # multi-step options
    multi_step_options = create_options(env)
    
    # track of steps per episode
    steps = np.zeros((N_EPISODES,))

    # total of 12 options: 4 primitive actions and 8 multi-step options
    # average Q_table
    avg_Q_table = np.zeros((13, 13, 12))

    for j in range(N_RUNS):
        Q_table = np.zeros((13, 13, 12))
        for i in range(N_EPISODES):
            done = False
            state = env.reset()
            current_steps = 0
            while not done:
                x, y = state
                option = policy(state, multi_step_options, Q_table)
                if intra_option_learning:
                    new_state, r, done, k = step(env, option, state, multi_step_options, Q_table, alpha, "intra option learning")
                else:
                    new_state, r, done, k = step(env, option, state, multi_step_options)
                # check valid options in new state
                valid_opts = valid_options(new_state, multi_step_options)
                x_new, y_new = new_state
                # Q - learning update
                Q_table[x, y, option] += alpha*(r + (env.gamma**k)*np.max(Q_table[x_new, y_new, valid_opts]) - Q_table[x, y, option])
                # print(state,"->", option, "->", new_state, "r:", r)
                state = new_state

                current_steps += k
                # print(current_steps)

            steps[i] += current_steps
        
        avg_Q_table +=  Q_table
            
        print("Run = ", j)
    return steps/N_RUNS, avg_Q_table/N_RUNS, multi_step_options


# composite step function to include multi-step options
def step(env, option, state, multi_step_options, Q_table=None, alpha=None, algorithm="SMDP Learning"):
    # for multi-step option
    if option > 3:
        # retrieve multi-step option
        option = multi_step_options[(option-4)[0]]
        # SMDP Q-learning 
        if algorithm == "SMDP Learning":
            state, r, done, k = option.run(state)
        # intra option Q-learning
        if algorithm == "intra option learning":
            state, r, done, k = option.run(state, "intra option learning", multi_step_options, Q_table, alpha)
    else:
        # primitive action
        state, r, done, _ = env.step(option)
        k = 1
    
    return state, r, done, k


# policy to choose a valid option given a state
def policy(state, multi_step_options, Q_table):
    x, y = state
    valid_opts = valid_options(state, multi_step_options)

    # epsilon - greedy option selection
    eps = 0.1
    if np.random.rand() < eps:
        # choose randomly from valid options
        option = np.random.choice(valid_opts, size = 1)
    else:
        # choose greedily from valid options
        option = greedy_option(x, y, valid_opts, Q_table)

    return option

def greedy_option(x, y, valid_opts, Q_table):
    # get all options with same max value
    max_equal_opts = []
    max_Q = np.max(Q_table[x, y, valid_opts])
    for o in valid_opts:
        if Q_table[x, y, o] == max_Q:
            max_equal_opts.append(o)

    return np.random.choice(max_equal_opts, size=1)

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

# create all 8 multi-step options
def create_options(env):
    HallwayOption.env = env
    # hallways are of two types, 0 -> walls on left and right
    # 1 -> walls on top and bottom
    options = []
    for i in range(4):
        options.append(HallwayOption(0, i+1))
        options.append(HallwayOption(1, i+1))
    
    return options

if __name__ == '__main__':
    # 1 -> G1, 2 -> G2
    goal = 2
    initial_state_fix = False
    intra_option_learning = True

    main(goal, initial_state_fix, intra_option_learning)