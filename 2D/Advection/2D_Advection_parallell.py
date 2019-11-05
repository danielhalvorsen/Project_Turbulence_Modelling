import numpy as np
import matplotlib.pyplot as plt

###########
# Read in velocity field files
dt_list = np.load('/home/danieloh/PycharmProjects/Project_Turbulence_Modelling/2D'
                  '/vorticity_formulation/datafiles/tlist.npy')
u_vel = np.load('/home/danieloh/PycharmProjects/Project_Turbulence_Modelling/2D'
                '/vorticity_formulation/datafiles/u_vel.npy')
v_vel = np.load('/home/danieloh/PycharmProjects/Project_Turbulence_Modelling/2D'
                '/vorticity_formulation/datafiles/v_vel.npy')
vorticity = np.load(
    '/home/danieloh/PycharmProjects/Project_Turbulence_Modelling/2D'
    '/vorticity_formulation/datafiles/omega.npy')
print('-----------Loaded in text files-----')
print('Vorticity shape: ', np.shape(vorticity))
L = np.pi
N = int((np.shape(u_vel[0, :]))[0])
dx = 2 * L / N
dy = dx
x = np.linspace(1 - N / 2, N / 2, N) * dx
y = np.linspace(1 - N / 2, N / 2, N) * dx
[X, Y] = np.meshgrid(x, y)

time_levels = int(len(dt_list))
dt = dt_list[1] - dt_list[0]
t_end = dt_list[-1]

S = np.zeros([N, N])
S_new = S.copy()
res = S.copy()
# S[int(N / 2) - int(N / 10):int(N / 2) + int(N / 10),
# int(N / 2) - int(N / 10):int(N / 2) + int(N / 10)] = 1
S = np.exp(-4 * np.log(2) * ((X) ** 2 + (Y) ** 2) / 3 ** 2)
init_mass = np.sum(np.sum(S))
fig, axs = plt.subplots(2)
fig.suptitle('Title here')
# ax = plt.axes(xlim=(0,N),ylim=(0,N))
# domain, = ax.plot(S)

counter = 0
scheme = 'First_Upwind'
if scheme == 'Second_Upwind':
    for t in range(time_levels):
        # Temporal loop
        for i in range(N):
            # Spatial loop, x-direction
            for j in range(N):
                # Spatial loop, y-direction
                # TODO incorporate these variables into a function
                u_plus = np.max(u_vel[t, i, j], 0)
                u_min = np.min(u_vel[t, i, j], 0)
                v_plus = np.max(v_vel[t, i, j], 0)
                v_min = np.min(v_vel[t, i, j], 0)
                Sx_plus = (-S[(i + 2) % N, j] + 4 * S[(i + 1) % N, j] - 3 * S[i, j]) / (
                        2 * dx)
                Sx_min = (3 * S[i, j] - 4 * S[i - 1, j] + S[i - 2, j]) / (2 * dx)
                Sy_plus = (-S[i, (j + 2) % N] + 4 * S[i, (j + 1) % N] - 3 * S[i, j]) / (
                        2 * dy)
                Sy_min = (3 * S[i, j] - 4 * S[i, j - 1] + S[i, j - 2]) / (2 * dy)

                res[i, j] = -(u_plus * Sx_min + u_min * Sx_plus) - (
                        v_plus * Sy_min + v_min * Sy_plus)
        S_new = S + dt * res
        #   domain.set_data(S)
        S = S_new.copy()
        max_u = np.max(np.abs(u_vel))
        max_v = np.max(np.abs(v_vel))
        max_vel = np.max([max_u, max_v])
        cfl = np.abs(max_vel * dt / dx)
        mass = np.sum(np.sum(S))
        print('Mass fraction: ', mass / init_mass)
        print('Max CFL value: ', cfl, '    Time level:  ', dt_list[t])
        # TODO VALUES OF SEDIEMTN OSCILLATES BETWEEN POSITIVE ANG NEGATIVE, WHY??
        if counter % 10 == 0:
            plt.suptitle(
                'Max CFL value: ' + np.str(cfl) + '   Time level:  ' + str(dt_list[t]))
            axs[0].imshow(S.T, cmap='jet', vmin=0, vmax=1)
            axs[1].imshow(np.abs((u_vel[t] ** 2) + (v_vel[t] ** 2)), cmap='jet')
            axs[1].quiver(u_vel[t], v_vel[t])
            plt.pause(0.005)
        counter += 1
    plt.show()

if scheme == 'First_Upwind':
    for t in range(time_levels):
        # Temporal loop
        for i in range(N):
            # Spatial loop, x-direction
            for j in range(N):
                # Spatial loop, y-direction
                # TODO incorporate these variables into a function
                u_plus = np.max(u_vel[t, i, j], 0)
                u_min = np.min(u_vel[t, i, j], 0)
                v_plus = np.max(v_vel[t, i, j], 0)
                v_min = np.min(v_vel[t, i, j], 0)
                Sx_plus = (S[(i + 1) % N, j] - S[i, j]) / dx
                Sx_min = (S[i, j] - S[i - 1, j]) / dx
                Sy_plus = (S[i, (j + 1) % N] - S[i, j]) / dy
                Sy_min = (S[i, j] - S[i, j - 1]) / dy

                res[i, j] = -(u_plus * Sx_min + u_min * Sx_plus) - (
                        v_plus * Sy_min + v_min * Sy_plus)
        S_new = S + dt * res
        #   domain.set_data(S)
        S = S_new.copy()
        max_u = np.max(np.abs(u_vel))
        max_v = np.max(np.abs(v_vel))
        max_vel = np.max([max_u, max_v])
        cfl = np.abs(max_vel * dt / dx)
        mass = np.sum(np.sum(S))
        print('Mass fraction: ', mass / init_mass)
        print('Max CFL value: ', cfl, '    Time level:  ', dt_list[t])
        # TODO VALUES OF SEDIEMTN OSCILLATES BETWEEN POSITIVE ANG NEGATIVE, WHY??
        if counter % 10 == 0:
            plt.suptitle(
                'Max CFL value: ' + np.str(cfl) + '   Time level:  ' + str(
                    dt_list[t]))
            axs[0].imshow(S.T, cmap='jet', vmin=0, vmax=1)
            axs[1].imshow(np.abs((u_vel[t] ** 2) + (v_vel[t] ** 2)), cmap='jet')
            axs[1].quiver(u_vel[t], v_vel[t])
            plt.pause(0.005)
        counter += 1
    plt.show()

'''     M = np.hypot(u_vel[t], v_vel[t])
        Q = axs[1].quiver(X, Y, u_vel[t], v_vel[t], M, units='x', 
        pivot='tip',
                          width=0.022,
                       scale=1 / 0.15)
        qk = axs[1].quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', 
        labelpos='E',
                           coordinates='figure')
        axs[1].scatter(X, Y, color='0.5', s=1)
'''
