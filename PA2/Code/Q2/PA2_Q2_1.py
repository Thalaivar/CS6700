import gym
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

# (alpha, gamma, N, iter)
# 0.25, 0.1, 500, 100
# 0.025, 0.7, 500, 200
# 0.025, 0.9, 500, 200

# lower gamma not good
# larger step size ruins things

ACTION_BOUNDS = 0.025
STATE_DIM = 2
ACTION_DIM = 2

def main():
    alpha = 0.025
    N = 500
    gamma = 0.3
    alpha_w = 0.0001

    env = gym.make("PA2_envs:vishamC-v0")
    env.tol = 1e-3

    avg_rewards, theta = train(500, env, [N, alpha, gamma, alpha_w])

    np.save("no_baseline_pg_visham_results", avg_rewards)
    np.save("no_baseline_pg_visham_policy", theta)

    plt.plot(np.arange(len(avg_rewards)), avg_rewards)
    plt.show()

def train(iterations, env, params):
    N, alpha, gamma, alpha_w = params

    # randomly initialize parameters
    theta = np.random.randn(2, 3)
    # w = np.random.randn(6,1)

    avg_rewards = np.zeros((iterations,))
    for i in range(iterations):
        reward = 0
        grad = np.zeros((ACTION_DIM, STATE_DIM + 1))

        # generate N trajectories and accumulate gradient
        for j in range(N):
            s_0 = env.reset()
            s_0 = np.append(s_0, 1)

            # generate trajectory
            traj = generate_trajectory(s_0, theta, env)
            # print(traj[1])
            returns = calculate_returns(traj[3], gamma)

            # train baseline
            # w = train_baseline(returns, traj[1], w, alpha_w)

            # accummulate gradient over trajectory
            dgrad = accumulate_grad(traj, theta, baseline, returns)
            grad += dgrad
            reward += np.sum(traj[3])

            # print("iter: %d ; trajectory: %d" % (i, j))

        # grad = grad/N
        avg_rewards[i] = reward/N

        print("iter: %d" % (i))

        # normalize grad
        grad = grad/(norm(grad) + 1e-8)

        # update theta
        theta += alpha*grad
        # alpha -= 1e-4

    return avg_rewards, theta

def baseline_basis(x):
    return np.array([[x[0]], [x[1]], [1], [x[0]**2], [x[1]**2], [x[1]*x[0]]])

def baseline(s, w):
    phi = baseline_basis(s)
    return (phi.T).dot(w), phi

def train_baseline(returns, states, w, alpha_w):
    # fit baseline to samples from trajectory
    steps = np.size(returns)
    
    # policy evaluation
    for i in range(steps):
        v_cap, phi = baseline(states[i], w)
        w += alpha_w*(returns[i] - v_cap)*phi

    return w

def accumulate_grad(trajectory, theta, baseline, returns, w = None):
    steps, states, actions, _ = trajectory

    grad = np.zeros((ACTION_DIM, STATE_DIM + 1))
    for i in range(steps):
        # b, _ = baseline(states[i], w)
        b = 0
        grad += policy_gradient(theta, states[i], actions[i])*(returns[i] - b)
    
    return grad

def calculate_returns(rewards, gamma):
    steps = np.size(rewards)

    G = np.zeros((steps,))
    
    for i in range(steps):
        Gt = 0
        j = 0
        for r in rewards[i:]:
            Gt += (gamma**j)*r
            j += 1

        G[i] = Gt
    
    return G

def generate_trajectory(s_0, theta, env):
    MAX_STEPS = 40
    # r(t)
    rewards = []
    # a(t)
    actions = []
    # s(t)
    states  = []

    done = False
    steps = 0
    state = s_0
    while not done and steps < MAX_STEPS:
        # record state
        states.append(state)

        # take step
        a = policy(theta, state)
        
        new_state, r, done, _ = env.step(a)

        state = np.append(new_state, 1)

        # record action and reward
        actions.append(a)
        rewards.append(r)

        steps += 1

    trajectory = [steps, np.array(states), np.array(actions), np.array(rewards)]
    return trajectory

def policy_gradient(theta, s, a):
    s = np.reshape(s, (STATE_DIM+1,1))
    # weights corresponding to the two actions
    theta_1 = np.reshape(theta[0,:], (STATE_DIM+1,1))
    theta_2 = np.reshape(theta[1,:], (STATE_DIM+1,1))

    grad_1 = (a[0] - (theta_1.T).dot(s))*(s.T)
    grad_2 = (a[1] - (theta_2.T).dot(s))*(s.T)
    grad = np.concatenate((grad_1, grad_2), axis=0)

    return grad

def policy(theta, s):
    s = np.reshape(s, (STATE_DIM+1,1))
    mean = theta.dot(s)
    # sample from normal
    a = np.random.normal(loc=mean, scale=1, size=(2,1))
    # normalize within bounds
    if norm(a) > ACTION_BOUNDS:
        a = ACTION_BOUNDS*a/norm(a)

    return a

if __name__ == "__main__":
    main()    