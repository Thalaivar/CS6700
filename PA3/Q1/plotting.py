import gym
import numpy as np
from options import HallwayOption
import matplotlib.pyplot as plt
from matplotlib import colors
from main import create_options, valid_options

def visualize(Q_table, env, steps, options=None):

    plt.plot(np.arange(1000), steps)
    plt.xlabel("Episodes")
    plt.ylabel("Steps")
    plt.show()

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

if __name__ == "__main__":
    env = gym.make("room_world:room-v0")

    """
    SMDP Q-Learning with goal as G1
    """
    # Q_table = np.load("results/results_Qtable_G1.npy")
    # steps = np.load("results/results_steps_G1.npy")
    # visualize(Q_table, env, steps)

    """
    SMDP Q-Learning with goal as G2
    """
    # Q_table = np.load("results/results_Qtable_G2.npy")
    # steps = np.load("results/results_steps_G2.npy")
    # visualize(Q_table, env, steps)
    
    """
    Part 2 with initial state fixed (goal G1)
    """
    # Q_table = np.load("results/results_Qtable_Q2_G1.npy")
    # steps = np.load("results/results_steps_Q2_G1.npy")
    # visualize(Q_table, env, steps)

    """
    Part 2 with initial state fixed (goal G2)
    """
    # Q_table = np.load("results/results_Qtable_Q2_G2.npy")
    # steps = np.load("results/results_steps_Q2_G2.npy")
    # visualize(Q_table, env, steps)

    """
    Intra option learning with G1
    """
    Q_table = np.load("results/results_Qtable_IO_G1.npy")
    steps = np.load("results/results_steps_IO_G1.npy")
    visualize(Q_table, env, steps)