import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Spatial constants
N = int(128)
dx = 1 / N;
dy = 1 / N;
L = 2 * np.pi
x = np.linspace(0, L, N + 1)
y = np.linspace(0, L, N + 1)
[X, Y] = np.meshgrid(x, y)
nu = 1e-2
# Temporal constants
tend = 10
dt = 0.0000001
n = int(np.ceil(tend / dt))

# Temporary velocity field. Steady state
u = np.cos(X) * np.sin(Y)*0.1
v = u.copy()

# initial sediment concentration
pos = np.dstack((X, Y))
mu = np.array([2, 3])
cov = np.array([[.5, .10], [.10, .5]])
rv = multivariate_normal(mu, cov)
S = rv.pdf(pos)
S_old = S.copy()
S_new = S.copy()

Bx = nu * dt / (dx ** 2)
By = nu * dt / (dy ** 2)
alpha = 2*Bx+2*By

for t in range(n):
    for i in range(N):
        for j in range(N):
            Ax = u[i, j] * dt / dx
            Ay = v[i, j] * dt / dy

            ux = u[(i + 1) % N, j] - u[i - 1, j] / (2 * dx)
            vy = v[i, (j + 1) % N] - v[i, j - 1] / (2 * dy)
            E = (ux + vy) * dt
            beta = Ax+Ay
            stab = np.sqrt(alpha**2+beta**2)-alpha
            if stab > E:
                print('Stability criteria violated. stab: ',stab)


            S_new[i, j] = ((1 - 2 * Bx - 2 * By) / (1 + 2 * Bx + 2 * By)) * S_old[
                i, j] + ((-Ax + 2 * Bx) / (1 + 2 * Bx + 2 * By)) * S[(i + 1) % N, j] + (
                                      (Ax + 2 * Bx) / (1 + 2 * Bx + 2 * By)) * S[
                              i - 1, j] + ((-Ay + 2 * By) / (1 + 2 * Bx + 2 * By)) * S[i,(j + 1) %
                              N] + ((Ay + 2 * By) / (1 + 2 * Bx + 2 * By)) * S[i, j - 1] - ((2 * E) / (1 + 2 * Bx + 2 * By)) * S[i, j]

    S_old = S.copy()
    S     = S_new.copy()
    plt.imshow(S)
    plt.pause(0.05)
    print(E)

