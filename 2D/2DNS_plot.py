import pickle
import matplotlib.pyplot as plt
from numpy import *
from mayavi import mlab
from basic_units import radians, degrees, cos
from radians_plot import *
import matplotlib.animation as animation
import types

# X = mgrid[rank * Np:(rank + 1) * Np, :N, :N].astype(float) * 2 * pi / N
# U = empty((3, Np, N, N),dtype=float32)
with open('X_2D.pkl', 'rb') as g:
    X_2D = pickle.load(g)
with open('U_2D.pkl', 'rb') as f:
    U_2D = pickle.load(f)
#with open('animate_U_x.pkl', 'rb') as h:
#    animate_U_x_pkl = pickle.load(h)
print(U_2D)
print(np.shape(U_2D))
print(np.shape(X_2D))
#print(np.shape(animate_U_x_pkl))


X_x = X_2D[0][0]
X_y = X_2D[0][1]
U_x = U_2D[0][0]
U_y = U_2D[0][1]

print(np.shape(U_x))

# Plot contour lines of the velocity in X-direction in the middle of the cube.
# X mesh is listed by X([z-levels,],[y-levels],[x-levels]), addressing, X[2] points to
# the mesh in x-direction.
plt.contourf(X_x, X_y, U_x,
             xunits=radians, yunits=radians, levels=256, cmap=plt.get_cmap('jet'))
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
plt.show()


fig = plt.figure()
# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(0,len(animate_U_x_T),10):
    print('Appending image nr: '+str(i))
    im = plt.imshow(animate_U_x_T[i][mid_idx], animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                repeat_delay=None)

#ani.save('dynamic_images.mp4')
ani.save('animation.gif', writer='imagemagick', fps=30)

plt.show()