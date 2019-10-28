import numpy as np
import matplotlib.pyplot as plt

###########
# Read in velocity field files

dt_list = np.loadtxt('dt_vector.txt')
u_vel = np.loadtxt('u_vel.txt')
v_vel = np.loadtxt('v_vel.txt')
vorticity = np.loadtxt('vorticity.txt')
print('-----------Loaded in text files-----')

L = np.pi
N = int(np.sqrt(np.shape(u_vel[0, :]))[0])
dx = 2 * L / N
dy = dx

time_levels = int(len(dt_list))
dt = dt_list[1] - dt_list[0]
t_end = dt_list[-1]

## Reshape velocity data to 2D:
print(np.shape(u_vel))

u_vel = np.reshape(u_vel, ([time_levels, N, N]), 1)
v_vel = np.reshape(v_vel, ([time_levels, N, N]), 1)

# print(np.shape(u_vel[-1]))
# plt.imshow(np.abs((u_vel[1] ** 2) + (v_vel[-1] ** 2)), cmap='jet')
# plt.show()

S = np.zeros([N, N])
S_new = S.copy()
res = S.copy()
S[int(N / 2) - int(N / 10):int(N / 2) + int(N / 10),
int(N / 2) - int(N / 10):int(N / 2) + int(N / 10)] = 1

#fig = plt.figure()
#ax = plt.axes(xlim=(0,N),ylim=(0,N))
#domain, = ax.plot(S)
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
    print('Max CFL value: ', cfl,'    Time level:  ',dt_list[t])
# TODO VALUES OF SEDIEMTN OSCILLATES BETWEEN POSITIVE ANG NEGATIVE, WHY??
    plt.imshow(S,cmap='jet',vmin=0,vmax=1)
    plt.pause(0.005)
plt.show()
