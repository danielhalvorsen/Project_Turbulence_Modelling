import pickle
import matplotlib.pyplot as plt
from numpy import *
from mayavi import mlab
from basic_units import radians, degrees, cos
from radians_plot import *
import matplotlib.animation as animation

# X = mgrid[rank * Np:(rank + 1) * Np, :N, :N].astype(float) * 2 * pi / N
# U = empty((3, Np, N, N),dtype=float32)
with open('X.pkl', 'rb') as g:
    X_pkl = pickle.load(g)
with open('U.pkl', 'rb') as f:
    U_pkl = pickle.load(f)
with open('animate_U_x.pkl', 'rb') as h:
    animate_U_x_pkl = pickle.load(h)

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





N = int(len(X[0, 0, 0, :]))
mid_idx = int(N / 2)

#mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(U[0]),
#                                 plane_orientation='z_axes',
#                                 slice_index=mid_idx,
#                                 )
#mlab.axes(xlabel='x', ylabel='y', zlabel='z')
#mlab.outline()
# mlab.show()


# Plot contour lines of the velocity in X-direction in the middle of the cube.
# X mesh is listed by X([z-levels,],[y-levels],[x-levels]), addressing, X[2] points to
# the mesh in x-direction.
plt.contourf(X[2, 0], X[1, 0], U_x[mid_idx],
             xunits=radians, yunits=radians, levels=30, cmap=plt.get_cmap('jet'))
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
#plt.show()


fig = plt.figure()
