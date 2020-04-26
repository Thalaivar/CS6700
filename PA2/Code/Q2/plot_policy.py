import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

ACTION_BOUNDS = 0.025
STATE_DIM = 2

# load policy parameters
theta = np.load("no_baseline_pg_visham_policy.npy")

# construct grid for plotting policy
N = 25

vals = np.linspace(-1, 1, N)
X, Y = np.meshgrid(vals, vals)

def policy(theta, s):
    s = np.reshape(s, (STATE_DIM+1,1))
    mean = theta.dot(s)
    # sample from normal
    a = np.random.normal(loc=mean, scale=1, size=(2,1))
    # normalize within bounds
    if norm(a) > ACTION_BOUNDS:
        a = ACTION_BOUNDS*a/norm(a)

    return a

# to generate policy outputs
def policy_output(X, Y, theta):
    AVG_TIMES = 50
    policy_x = np.zeros(X.shape)
    policy_y = np.zeros(Y.shape)

    for k in range(AVG_TIMES):
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                x = X[i,j]; y = Y[i,j]
                a = policy(theta, [x, y, 1])
                policy_x[i,j] += a[0,0]
                policy_y[i,j] += a[1,0]

    return policy_x/AVG_TIMES, policy_y/AVG_TIMES

policy_x, policy_y = policy_output(X, Y, theta)

plt.quiver(X, Y, policy_x, policy_y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
