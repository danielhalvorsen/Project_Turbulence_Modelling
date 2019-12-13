import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

N = 64
dt = 0.01
tend = 50
L = 2 * np.pi
dx = L / N
x = np.arange(0, N, 1) * L / N
y = np.arange(0, N, 1) * L / N
[X, Y] = np.meshgrid(x, y)
n = int(np.ceil(tend / dt))

u = np.ones((N, N)) * 1
v = np.ones((N, N)) * 1

#u = np.sin(X) * np.cos(Y) * 0.1
#v = np.cos(X) * np.cos(Y) * 0.1

# u = np.random.rand(N, N)*1
# v = np.random.rand(N, N)*1

D = 0.08
r = dt * D / (dx ** 2)
cell_Reynold = np.max(u) / N / D
Peclet = np.max(u) / N / D
print('Peclet: ', Peclet, ' 2*von Neumann: ', 2 * r)
# assert Peclet <=2

# plt.imshow(u)
# plt.show()

pos = np.dstack((X, Y))
mu = np.array([2, 3])
cov = np.array([[.05, .010], [.010, .05]])
rv = multivariate_normal(mu, cov)
S = rv.pdf(pos)
sol = S.copy()/(np.sum(S))
sol_new = S.copy()

# sol = np.zeros((N,N))
# sol[int(N/2),int(N/2)]=5
# sol_new = sol.copy()

# sol = np.cos(X)*np.sin(Y)
# plt.imshow(sol)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# Ln, = ax.plot(sol)
# ax.set_xlim([0, 2 * np.pi])
# plt.ion()1
# plt.show()

fig, axs = plt.subplots(2)
fig.suptitle('Title here')

for t in range(n):
    for i in range(len(x)):
        for j in range(len(x)):
            cu = dt * u[i, j] / dx
            cv = dt * v[i, j] / dx
            assert (((cu ** 2) / r) + ((cv ** 2) / r) <= 2),('dt might be too high or diffusion constant might be too low')
            assert (0 < 2 * r <= 0.5)
            sol_new[i, j] = sol[i, j] \
                            + cu / 2 * (sol[(i - 1), j] - sol[(i + 1) % N, j ]) \
                            + cv / 2 * (sol[i % N, j - 1] - sol[i % N, (j + 1) % N]) \
                            + r * (sol[i - 1, j ] - 2 * sol[i , j ] + sol[(i + 1) % N, j ]) \
                            + r * (sol[i , j - 1] - 2 * sol[i , j ] + sol[i , (j + 1) % N])
    sol = sol_new
    sol = sol/(np.sum(sol)) #cheat with mass conservation. Assume uniform loss over each cell

    #    plt.plot(x,sol_new)

    #    Ln.set_ydata(sol)
    #    Ln.set_xdata(X)
    if (t % 20 == 0):
        # plt.clf()
        print('time level: ', t, '              Total Sediment: ', np.sum(sol))
        # plt.imshow(sol)
        # plt.contour(sol)
        # plt.pause(0.005)
        #   plt.imshow(sol)
        #   plt.show()
        axs[0].imshow(sol, cmap='jet')  # ,vmax=1,vmin=0)
        axs[1].imshow(v, cmap='jet')
        plt.pause(0.05)
