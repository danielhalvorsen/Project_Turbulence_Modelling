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
with open('X.pkl', 'rb') as g:
    X_pkl = pickle.load(g)
with open('U.pkl', 'rb') as f:
    U_pkl = pickle.load(f)
with open('animate_U_x.pkl', 'rb') as h:
    animate_U_x_pkl = pickle.load(h)

print(np.shape(U_pkl))
print(np.shape(animate_U_x_pkl))


# Concatenates the processor axis on spatial mesh, X, and solution mesh, U. This is
# done in a dynamical for-loop which changes size depending on number of processors.
X_list = [X_pkl[0][i] for i in range(np.size(X_pkl, 1))]
U_list = [U_pkl[0][i] for i in range(np.size(U_pkl, 1))]
animate_U_x_list = [animate_U_x_pkl[0][i] for i in range(np.size(X_pkl, 1))]
X = concatenate(X_list, axis=1)
U = concatenate(U_list, axis=1)
animate_U_x = concatenate(animate_U_x_list, axis=1)

U_x = U[0].transpose((2, 1, 0))
U_y = U[1].transpose((2, 1, 0))
U_z = U[2].transpose((2, 1, 0))
animate_U_x_T = animate_U_x.transpose((0,3,2,1))


print(np.shape(U_x))
print(np.shape(animate_U_x_T))

N = int(len(X[0, 0, 0, :]))
mid_idx = int(N / 2)

mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(U[0]),
                                 plane_orientation='z_axes',
                                 slice_index=mid_idx,
                                 )
mlab.axes(xlabel='x', ylabel='y', zlabel='z')
mlab.outline()
# mlab.show()


# Plot contour lines of the velocity in X-direction in the middle of the cube.
# X mesh is listed by X([z-levels,],[y-levels],[x-levels]), addressing, X[2] points to
# the mesh in x-direction.
plt.contourf(X[2, 0], X[1, 0], U_x[mid_idx],
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


with open('animate_U_x_'+str(0)+'.pkl', 'rb') as i:
    U_0= pickle.load(i)
with open('animate_U_x_'+str(1)+'.pkl', 'rb') as j:
    U_1= pickle.load(j)
with open('animate_U_x_'+str(2)+'.pkl', 'rb') as k:
    U_2= pickle.load(k)
with open('animate_U_x_'+str(3)+'.pkl', 'rb') as l:
    U_3= pickle.load(l)
print(np.shape(U_0))

U_anim = concatenate([U_0,U_1,U_2,U_3],axis=2)

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