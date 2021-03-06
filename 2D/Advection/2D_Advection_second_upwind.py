import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numba import jit
###########
# Read in velocity field files

dt_list = np.load('dt_vector.npy')
u_vel = np.load('u_vel.npy')
v_vel = np.load('v_vel.npy')
vorticity = np.load('vorticity.npy')
print('-----------Loaded in text files-----')
print('Vorticity shape: ', np.shape(vorticity))
L = np.pi
N = int(np.sqrt(np.shape(u_vel[0, :]))[0])
dx = 2 * L / N
dy = dx
x = np.linspace(1 - N / 2, N / 2, N) * dx
y = np.linspace(1 - N / 2, N / 2, N) * dx
[X, Y] = np.meshgrid(x, y)

time_levels = int(len(dt_list))
dt = dt_list[1] - dt_list[0]
t_end = dt_list[-1]

## Reshape velocity data to 2D:
print('U-velocity shape:', np.shape(u_vel))

u_vel = np.reshape(u_vel, ([time_levels, N, N]), 1)
v_vel = np.reshape(v_vel, ([time_levels, N, N]), 1)

# print(np.shape(u_vel[-1]))
# plt.imshow(np.abs((u_vel[1] ** 2) + (v_vel[-1] ** 2)), cmap='jet')
# plt.show()

S = np.zeros([N, N])
S_new = S.copy()
res = S.copy()
#S[int(N / 2) - int(N / 10):int(N / 2) + int(N / 10),
#int(N / 2) - int(N / 10):int(N / 2) + int(N / 10)] = 1




pos = np.dstack((X, Y))
mu = np.array([1, 2])
cov = np.array([[.5, .25],[.25, .5]])
rv = multivariate_normal(mu, cov)
S = rv.pdf(pos)
print('Initial Sediment: ',np.sum(np.sum(S)))

fig,axs = plt.subplots(2)
fig.suptitle('Title here')
#ax = plt.axes(xlim=(0,N),ylim=(0,N))
#domain, = ax.plot(S)

scheme = 'Second_Upwind'
if scheme=='Second_Upwind':
    for t in range(time_levels):
        # Temporal loop
        for i in range(N):
            # Spatial loop, x-direction
            for j in range(N):
                # Spatial loop, y-direction
                #TODO incorporate these variables into a function
                u_plus  = np.max(u_vel[t, i, j], 0)
                u_min   = np.min(u_vel[t, i, j], 0)
                v_plus  = np.max(v_vel[t, i, j], 0)
                v_min  = np.min(v_vel[t, i, j], 0)
                Sx_plus = (-S[(i+2)%N,j]+4*S[(i+1)%N,j]-3*S[i,j])/(2*dx)
                Sx_min  = (3*S[i,j]-4*S[i-1,j]+S[i-2,j])/(2*dx)
                Sy_plus = (-S[i,(j+2)%N]+4*S[i,(j+1)%N]-3*S[i,j])/(2*dy)
                Sy_min  = (3*S[i,j]-4*S[i,j-1]+S[i,j-2])/(2*dy)

                res[i,j] = -(u_plus*Sx_min+u_min*Sx_plus)-(v_plus*Sy_min+v_min*Sy_plus)
        S_new = S+dt*res
     #   domain.set_data(S)
        S = S_new.copy()
        max_u = np.max(np.abs(u_vel))
        max_v = np.max(np.abs(v_vel))
        max_vel = np.max([max_u,max_v])
        cfl = np.abs(max_vel*dt/dx)
        print('Max CFL value: ', cfl,'    Time level:  ',dt_list[t],'            Total Sediment: ', np.sum(np.sum(S)))
    # TODO VALUES OF SEDIEMTN OSCILLATES BETWEEN POSITIVE ANG NEGATIVE, WHY??

        plt.suptitle('Max CFL value: '+np.str(cfl)+'   Time level:  '+str(dt_list[t]))
        axs[0].imshow(S.T,cmap='jet',vmin=0,vmax=1)
        axs[1].imshow(np.abs((u_vel[t] ** 2) + (v_vel[t] ** 2)), cmap='jet')
        plt.pause(0.005)
    plt.show()


'''     M = np.hypot(u_vel[t], v_vel[t])
        Q = axs[1].quiver(X, Y, u_vel[t], v_vel[t], M, units='x', pivot='tip',
                          width=0.022,
                       scale=1 / 0.15)
        qk = axs[1].quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                           coordinates='figure')
        axs[1].scatter(X, Y, color='0.5', s=1)
'''

