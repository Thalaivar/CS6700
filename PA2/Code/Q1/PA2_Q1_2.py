import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# goal states
A = (11,11)
B = (9,9)
C = (7,5)

gamma = 0.9

def main(goal_state, wind, distance_reward):
    env = gym.make("puddle_world:puddle-v0")
    # set problem parameters
    env.set_goal_state(goal_state)
    env.set_wind(wind)
    env.set_distance_reward(distance_reward)

    rewards, steps, Q_table = train(50, 1000, env)

    # np.save("Rewards", rewards)

    plot_stats(rewards, steps)
    plot_policy(Q_table, env)

def train(runs, episodes, env):
    # average rewards and steps
    rewards = np.zeros((episodes,))
    steps = np.zeros((episodes,))

    # Q-table for run with best reward
    Q_table_max = np.zeros((12,12,4))

    # to check for best run
    max_run_reward = None

    run = 0
    while run < runs:
        Q_table = np.zeros((12,12,4))

        # check max reward recieved in this run
        max_episode_reward = None

        for j in range(episodes):
            done = False
            state = env.reset()

            # reward and steps for current episode
            current_reward = 0
            current_steps = 0

            # take first action
            action = derive_action(Q_table, state, env)
            while not done:
                new_state, r, done, _ = env.step(action)

                Q_table, new_action = SARSA_update(state, new_state, action, r, Q_table, env)

                # update quantities
                action = new_action
                state = new_state
                current_reward += r
                current_steps += 1
            
            # update rewards and steps
            rewards[j] += current_reward
            steps[j] += current_steps

            if max_episode_reward is None or current_reward > max_episode_reward:
                max_episode_reward = current_reward
        
        # save the Q-table that gets the best reward across all episodes and runs
        if max_run_reward is None or max_episode_reward > max_run_reward:
            max_run_reward = max_episode_reward
            Q_table_max = Q_table

        print("Run: %d" % (run))
        run += 1
    
    return rewards/runs, steps/runs, Q_table_max


# ε-greedy policy
def derive_action(Q_table, s, env):
    eps = 0.01
    if np.random.rand() < eps:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q_table[s[0], s[1], :])

    return action

# SARSA update
def SARSA_update(s, new_s, a, r, Q_table, env):
    alpha = 0.5

    Q_prev = Q_table[s[0], s[1], a]
    
    # derive new action
    new_a = derive_action(Q_table, new_s, env)
    Q_new = Q_table[new_s[0], new_s[1], new_a]

    Q_table[s[0], s[1], a] = Q_prev + alpha*(r + gamma*Q_new - Q_prev)

    return Q_table, new_a

# plot policy
def plot_policy(Q_table, env):
    gridworld = np.zeros((12,12))
    
    for s in env.DEPTH_1:
        gridworld[s[0],s[1]] = 1
    for s in env.DEPTH_2:
        gridworld[s[0],s[1]] = 2
    for s in env.DEPTH_3:
        gridworld[s[0],s[1]] = 3
    
    goal_state = env.goal_state
    gridworld[goal_state[0], goal_state[1]] = 4

    gridworld = gridworld.T

    # plot puddle world
    cmap = colors.ListedColormap([[1,1,1],[0.631, 0.631, 0.631],[0.4, 0.4, 0.4], [0.168, 0.168, 0.168], [0.262, 0.741, 0.058]])
    plt.figure(figsize=(6,6))
    plt.pcolor(gridworld, cmap=cmap, edgecolors='k', linewidths=1)
    
    # plot actions
    space = np.arange(12, dtype="float64")
    X, Y = np.meshgrid(space, space)
    actions_x = np.zeros(gridworld.shape)
    actions_y = np.zeros(gridworld.shape)
    for i in range(12):
        for j in range(12):
            x = X[i,j]; y = Y[i,j]
            actions_x[i,j], actions_y[i,j] = map_actions(np.argmax(Q_table[int(x),int(y),:]))
    
    X += 0.5; Y += 0.5
    plt.quiver(X, Y, actions_x, actions_y)

    plt.show()

def plot_stats(rewards, steps):
    episodes = np.size(rewards)
    plt.plot(np.arange(episodes), rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.show()

    plt.plot(np.arange(episodes), steps)
    plt.xlabel("Episode")
    plt.ylabel("Average Steps")
    plt.show() 

def map_actions(action):
    if action == 0:
        return 0, 1
    elif action == 1:
        return 0, -1
    elif action == 2:
        return 1, 0
    elif action == 3:
        return -1, 0

if __name__ == "__main__":
    # wind on (True)/ off (False)
    wind = True
    # dense distance reward on (True)/ off (False)
    distance_reward = True

    main(C, wind, distance_reward)