import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter

N = 1000
vals = np.linspace(-1, 1, N)
X, Y = np.meshgrid(vals, vals)
Z = np.zeros(X.shape)

w = np.load("VFA_params_vishamC.npy")

def baseline_basis(x):
    return np.array([[x[0]], [x[1]], [1], [x[0]**2], [x[1]**2], [x[1]*x[0]]])

def baseline(s, w):
    phi = baseline_basis(s)
    return (phi.T).dot(w), phi

for i in range(N):
    for j in range(N):
        x = X[i,j]; y = Y[i,j]
        Z[i,j], _ = baseline(np.array([x,y]), w)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='summer', edgecolors='none')

ax.xaxis.set_major_locator(LinearLocator(5))
ax.yaxis.set_major_locator(LinearLocator(5))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"$\hat{V}(s, w)$")
plt.show()