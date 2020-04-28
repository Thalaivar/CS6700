import gym
import numpy as np
from options import HallwayOption
import matplotlib.pyplot as plt
from matplotlib import colors

N_EPISODES = 1000
N_RUNS = 100

def main(goal_state):
    env = gym.make("room_world:room-v0")

    if goal_state == 1:
        env.goal = env.G1
    elif goal_state == 2:
        env.goal = env.G2

    steps, Q_table, _ = train(env)
    plt.plot(np.arange(N_EPISODES), steps)
    plt.show()

    np.save("results_Qtable_G2", Q_table)
    np.save("results_steps_G2", steps)

    visualize(Q_table, env)

def train(env):
    alpha = 1/8

    # total of 12 options: 4 primitive actions and 8 multi-step options

    # multi-step options
    multi_step_options = create_options(env)
    
    # track of steps per episode
    steps = np.zeros((N_EPISODES,))
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
                new_state, r, done, k = step(env, option, state, multi_step_options)

                # check valid options in new state
                valid_opts = valid_options(state, multi_step_options)
                x_new, y_new = new_state
                # Q - learning update
                Q_table[x, y, option] += alpha*(r + (env.gamma**k)*np.max(Q_table[x_new, y_new, valid_opts]) - Q_table[x, y, option])
                # print(state,"->", option, "->", new_state, "r:", r)
                state = new_state

                current_steps +=1
                # print(current_steps)

            steps[i] += current_steps
        
        avg_Q_table +=  Q_table
            
        print("Run = ", j)
    return steps/N_RUNS, avg_Q_table/N_RUNS, multi_step_options

# composite step function to include multi-step options
def step(env, option, state, multi_step_options):
    # for multi-step option
    if option > 3:
        # retrieve multi-step option
        option = multi_step_options[(option-4)[0]]
        state, r, done, k = option.run(state)
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

def visualize(Q_table, env, options=None):
    if options is None:
        options = create_options(env)

    grid_size = [Q_table.shape[0], Q_table.shape[1]]
    # plot where options are taken
    vis_Q_table = np.zeros((grid_size[0], grid_size[1]))
     
    # primitive_track = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            state = [i,j]
            # mark walls in room world
            if state in env.walls:
                vis_Q_table[i,j] = -1
            else:
                # valid options in this state
                valid_opts = valid_options(state, options)
                # greedily choose best option
                option = np.argmax(Q_table[i,j,valid_opts])
                # differentiate between primitive actions and options
                if option > 3:
                    option = 1
                else:
                    # primitive_track.append([i,j,option])
                    option = 0
                vis_Q_table[i,j] = option

    vis_Q_table = vis_Q_table.T

    # colour code the results:
    # -1  -> walls
    # 0   -> primitive actions
    # 1   -> multi - step options
    cmap = colors.ListedColormap([[0.64,0.45,0.16], [0.17,0.35,0.62], [0.07,0.59,0.47]])
    plt.figure(figsize=(6,6))
    plt.pcolor(vis_Q_table, cmap=cmap, edgecolors='k', linewidths=1)
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    # legend stuff
    rect = lambda color: plt.Rectangle((0,0), 1, 1, color=color)
    plt.legend([rect([0.17,0.35,0.62]), rect([0.07, 0.59, 0.47])], ["primitive actions", "multi-step options"], loc='upper right')
    plt.show()

    # plot heatmap of max values
    val_vis_grid = np.zeros(vis_Q_table.shape)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            valid_opts = valid_options(state, options)
            val_vis_grid[i,j] = np.max(Q_table[i,j,valid_opts])

    plt.pcolormesh(val_vis_grid.T, cmap='viridis', linewidths=1, edgecolors='k')
    plt.colorbar()
    ax = plt.gca()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    empty_string_labels = ['']*len(labels)
    ax.set_xticklabels(empty_string_labels)
    ax.set_yticklabels(empty_string_labels)
    plt.show()

    # print(primitive_track)

if __name__ == '__main__':
    main(2)