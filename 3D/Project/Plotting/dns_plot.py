import pickle
import matplotlib.pyplot as plt
from numpy import *
from mayavi import mlab


#X = mgrid[rank * Np:(rank + 1) * Np, :N, :N].astype(float) * 2 * pi / N
#U = empty((3, Np, N, N),dtype=float32)
with open('X.pkl', 'rb') as g:
    X = pickle.load(g)
#Np=4
#N = int((2**6))
#U = empty((3, Np, N, N),dtype=float32)
#U = zeros(Np)
#for rank in range(Np):
with open('U_rank_'+str(0)+'.pkl', 'rb') as f:
    U_0= pickle.load(f)
with open('U_rank_'+str(1)+'.pkl', 'rb') as f:
    U_1= pickle.load(f)
with open('U_rank_'+str(2)+'.pkl', 'rb') as f:
    U_2= pickle.load(f)
with open('U_rank_'+str(3)+'.pkl', 'rb') as f:
    U_3= pickle.load(f)

#U=[U_0,U_1,U_2,U_3]
#U[0,:,:32,:,:]=U_0
U = concatenate([U_0,U_1,U_2,U_3],axis=2)
print(shape(U_0))
print(shape(U))

#print('printing X: \n\n')
##print(X[0][1][:,:,1])
#print('\n\nPrinting U: \n\n')
#print(U[0][0,:,:,3])
#print('\n\n')
#index pickle elements with [0]
#print(U[0] is float)
#plt.contourf(X[0][0][:, :, 0], X[0][0][:, :, 0], U[0][0,:, :, 0], 100)
#plt.colorbar()
#plt.show()

mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(U[0][2]),
                            plane_orientation='x_axes',
                            slice_index=100,
                    )
#mlab.quiver3d(U[0][0],U[0][1],U[0][2])

mlab.outline()
mlab.show()