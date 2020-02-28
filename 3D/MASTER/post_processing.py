import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ims = []
step=0
amount = 32
name = 'vel_files/velocity_'+str(step)+'.npy'

#TODO load in one and one file from /vel_files, read [0][:,:,-1] and add to animation. Also make spectrum plots and viscous diffusion plots
for i in range(amount):
    name = 'vel_files/velocity_' + str(step) + '.npy'
    vec = np.load(name)
    print('Loaded nr: '+str(step),flush=True)
    im = plt.imshow(vec[0][:,:,-1],cmap='jet', animated=True)
    ims.append([im])
    step += 140
    print('Finished appending nr: '+str(step),flush=True)



ani = animation.ArtistAnimation(fig, ims, interval=2, blit=True,repeat_delay=None)
ani.save('test.gif', writer='imagemagick')

