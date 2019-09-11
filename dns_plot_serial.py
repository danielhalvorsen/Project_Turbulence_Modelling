import pickle
import matplotlib.pyplot as plt
from numpy import *
from mayavi import mlab
from basic_units import radians,degrees,cos
from radians_plot import *

# TODO make plot script dynamical - change depending on # of cores used.

#X = mgrid[rank * Np:(rank + 1) * Np, :N, :N].astype(float) * 2 * pi / N
#U = empty((3, Np, N, N),dtype=float32)
with open('X.pkl', 'rb') as g:
    X = pickle.load(g)

#with open('U_rank_'+str(0)+'.pkl', 'rb') as f:
#    U= pickle.load(f)
with open('U.pkl', 'rb') as f:
    U= pickle.load(f)


X_new = concatenate([X[0][0],X[0][1],X[0][2],X[0][3]],axis=1)
U_new = concatenate([U[0][0],U[0][1],U[0][2],U[0][3]],axis=1)

mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(U_new[0]),
                            plane_orientation='x_axes',
                            slice_index=20,
                        )


mlab.outline()
#mlab.show()



N=int(len(X_new[0,0,0,:]))
L=1
mid_idx = int(N/2)

plt.contourf(X_new[0,:,:,mid_idx], X_new[1,:,:,mid_idx], U_new[0][:][:][mid_idx],xunits=radians,yunits=radians,levels=100)
ax = plt.gca()
ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
plt.show()
