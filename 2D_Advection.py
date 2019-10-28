import numpy as np
import matplotlib.pyplot as plt


###########
# Read in velocity field files

dt_list = np.loadtxt('dt_vector.txt')
u_vel = np.loadtxt('u_vel.txt')
v_vel = np.loadtxt('v_vel.txt')
vorticity = np.loadtxt('vorticity.txt')

N = int(np.sqrt(np.shape(u_vel[0,:]))[0])

time_levels = int(len(dt_list))
dt = dt_list[1]-dt_list[0]
t_end = dt_list[-1]

## Reshape velocity data to 2D:
print(np.shape(u_vel))


u_vel = np.reshape(u_vel,([time_levels,N, N]),1)
v_vel = np.reshape(v_vel,([time_levels,N, N]),1)

print(np.shape(u_vel[-1]))
plt.imshow(np.abs((u_vel[1] ** 2) + (v_vel[-1] ** 2)), cmap='jet')
plt.show()