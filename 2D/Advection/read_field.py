import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

tend = 100
dt = 1e-3
timesteps = int(np.ceil(tend/dt))
fig = plt.figure()
ims = []
for i in range(0,100):
    field = np.load('datafiles/concentrations/field_'+str(i*1000)+'.npy')
    im = plt.imshow(field,cmap='jet',animated=True)
    ims.append([im])
    print('saving field',i)
ani = animation.ArtistAnimation(fig, ims, interval=2, blit=True, repeat_delay=None)
ani.save('fieldspread.gif', writer='imagemagick')