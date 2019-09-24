import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import types

with open('solve_matrix.pkl', 'rb') as f:
    solve= pickle.load(f)
solve_matrix=solve[0]
max_val = solve_matrix.y.max()
min_val = solve_matrix.y.min()
print(max_val)
N=int(np.sqrt(len(solve_matrix.y[:,-1])))
#omega_vector = solve_matrix.y[:,-1]
#omega = np.reshape(omega_vector, ([N, N]))

#plt.contourf(omega,levels=500)

fig = plt.figure()
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(len(solve_matrix.y[0])):
    if i%3==0:
        print('Appending image nr: '+str(i))
        omega_vector = solve_matrix.y[:, i]
        omega = np.reshape(omega_vector, ([N, N]))
        #im = plt.contourf(omega,cmap='jet',vmax=max_val,vmin=min_val,levels=100,animated=True)
        im = plt.imshow(omega,animated=True,cmap='jet')
       # def setvisible(self, vis):
        #    for c in self.collections: c.set_visible(vis)
        #im.set_visible = types.MethodType(setvisible, im)
       # im.axes = plt.gca()
      #  im.figure = fig
        ims.append([im])

cbar = plt.colorbar(im)
cbar.set_clim(min_val,max_val)
cbar.set_ticks(np.linspace(min_val,max_val,10))
cbar.set_label('Vorticity magnitude [m/s]')
plt.xlim(0,N)
plt.ylim(0,N)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axes().set_aspect('equal')

ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                repeat_delay=None)

#ani.save('dynamic_images.mp4')
ani.save('animation.gif', writer='imagemagick')

plt.show()