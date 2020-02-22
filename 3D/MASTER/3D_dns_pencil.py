from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, fftn,fftshift,rfft,irfft
from mpi4py import MPI
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from mpistuff.mpibase import work_arrays,work_array_dict


# U is set to dtype float32
# Reynoldsnumber determined by nu Re = 1600, nu = 1/1600
work_array = work_arrays()
nu = 0.0000000625
# nu = 0.00000625
T = 40
dt = 0.01
L = 2*pi
N = int(2 ** 6)
N_half = int(N / 2)
N_nyquist=int(N/2+1)
P1 = 1
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
mpitype = MPI.DOUBLE_COMPLEX

if num_processes > 1:
    P1 = 2

rank = comm.Get_rank()

P2 = int(num_processes/P1)
N1 = int(N/P1)
N2 = int(N/P2)


commxz = comm.Split(rank/P1)
commxy = comm.Split(rank%P1)

xzrank = commxz.Get_rank()
xyrank = commxy.Get_rank()


x1 = slice(xzrank*N1,(xzrank+1)*N1,1)
x2 = slice(xyrank*N2,(xyrank+1)*N2,1)
X = mgrid[x1,x2,:N].astype(float)*L/N

kx = fftfreq(N, 1. / N)
kz = kx[:(N_half)].copy();
kz[-1] *= -1

k2 = slice(int(xyrank*N2),int((xyrank+1)*N2),1)
k1 = slice(int(xzrank*N1/2),int(xzrank*N1/2 + N1/2),1)
#k1 = slice(int(xzrank*N1/2),int((xzrank+1)*N1/2))
K = array(meshgrid(kx[k2],kx,kx[k1],indexing='ij'),dtype=int)


U = empty((3, N1, N2, N), dtype=float32)
U_hat = empty((3, N2, N, int(N_half/P1)), dtype=complex)
P = empty((N1, N2, N))
P_hat = empty((N2, N, int(N_half/P1)), dtype=complex)
U_hat0 = empty((3, N2, N, int(N_half/P1)), dtype=complex)
U_hat1 = empty((3, N2, N, int(N_half/P1)), dtype=complex)

Uc_hat = empty((N, N2, int(N_half/P1)), dtype=complex)
Uc_hatT = empty((N2, N, int(N_half/P1)), dtype=complex)
U_mpi = empty((num_processes, N1, N2, N_half), dtype=complex)

Uc_hat_x = empty((N, N2, int(N1/2)), dtype=complex)
Uc_hat_y = empty((N2, N, int(N1/2)), dtype=complex)
Uc_hat_z = empty((N1, N2, int(N_nyquist)), dtype=complex)
Uc_hat_xr = empty((N, N2, int(N1/2)), dtype=complex)


dU = empty((3, N2, N, int(N_half/P1)), dtype=complex)
curl = empty((3, N1, N2, N))

#K = array(meshgrid(kx, kx[rank * Np:(rank + 1) * Np], kz, indexing="ij"), dtype=int)
K2 = sum(K * K, 0, dtype=int)
K_over_K2 = K.astype(float) / where(K2 == 0, 1, K2).astype(float)
kmax_dealias = 2. / 3. * (N_half)
dealias = array(
    (abs(K[0]) < kmax_dealias) * (abs(K[1]) < kmax_dealias) * (abs(K[2]) < kmax_dealias),
    dtype=bool)

a = [1. / 6., 1. / 3., 1. / 3., 1. / 6.]
b = [0.5, 0.5, 1.]
#dir = '/home/danieloh/PycharmProjects/Project_Turbulence_Modelling/animation_folder/'

def ifftn_mpi2(fu, u):
    # Inverse Fourier transform
    Uc_hat[:] = ifft(fu, axis=0)
    comm.Alltoall([Uc_hat, MPI.DOUBLE_COMPLEX], [U_mpi, MPI.DOUBLE_COMPLEX])
    Uc_hatT[:] = rollaxis(U_mpi, 1).reshape(Uc_hatT.shape)
    u[:] = irfft2(Uc_hatT, axes=(1, 2))
    return u

def transform_Uc_zx(Uc_hat_z, Uc_hat_xr, P1):
    sz = Uc_hat_z.shape
    sx = Uc_hat_xr.shape
    Uc_hat_z[:, :, :-1] = rollaxis(Uc_hat_xr.reshape((P1, sz[0], sz[1], sx[2])), 0, 3).reshape((sz[0], sz[1], sz[2]-1))
    return Uc_hat_z

def transform_Uc_xy(Uc_hat_x, Uc_hat_y, P):
    sy = Uc_hat_y.shape
    sx = Uc_hat_x.shape
    Uc_hat_x[:] = rollaxis(Uc_hat_y.reshape((sy[0], P, sx[1], sx[2])), 1).reshape(sx)
    return Uc_hat_x


def ifftn_mpi(fu,u):
    Uc_hat_y = work_array[((N2, N, int(N1/2)),complex, 0, False)]
    Uc_hat_z = work_array[((N1, N2, N_nyquist), complex, 0, False)]

    Uc_hat_x = work_array[((N, N2, int(N1 / 2)), complex, 0, False)]
    Uc_hat_xp = work_array[((N, N2, int(N1/2)), complex, 0, False)]
    xy_plane = work_array[((N, N2), complex, 0, False)]
    xy_recv = work_array[((N1, N2), complex, 0, False)]

    # Do first owned direction
    Uc_hat_y = ifft(fu, axis=1)
    # Transform to x
    Uc_hat_xp = transform_Uc_xy(Uc_hat_xp, Uc_hat_y, P2)

    ###### In-place
    ## Communicate in xz-plane and do fft in x-direction
    # self.comm1.Alltoall(MPI.IN_PLACE, [Uc_hat_xp, self.mpitype])
    # Uc_hat_xp[:] = ifft(Uc_hat_xp, axis=0, threads=self.threads,
    # planner_effort=self.planner_effort['ifft'])

    # Uc_hat_x[:] = Uc_hat_xp[:, :, :self.N1[2]//2]

    ## Communicate and transform in xy-plane all but k=N//2
    # self.comm0.Alltoall(MPI.IN_PLACE, [Uc_hat_x, self.mpitype])

    ####### Not in-place
    # Communicate in xz-plane and do fft in x-direction
    Uc_hat_xp2 = work_array[((N, N2, int(N1/2)), complex, 1, False)]
    commxy.Alltoall([Uc_hat_xp, mpitype], [Uc_hat_xp2, mpitype])
    Uc_hat_xp = ifft(Uc_hat_xp2, axis=0)

    Uc_hat_x2 = work_array[((N, N2, int(N1 / 2)), complex, 1, False)]
    Uc_hat_x2[:] = Uc_hat_xp[:, :, :int(N1 / 2)]

    # Communicate and transform in xy-plane all but k=N//2
    commxz.Alltoall([Uc_hat_x2, mpitype], [Uc_hat_x, mpitype])
    #########################

    Uc_hat_z[:] = transform_Uc_zx(Uc_hat_z, Uc_hat_x, P1)

    xy_plane[:] = Uc_hat_xp[:, :, -1]
    commxz.Scatter(xy_plane, xy_recv, root=P1 - 1)
    Uc_hat_z[:, :, -1] = xy_recv

    # Do ifft for z-direction
    u = irfft(Uc_hat_z, axis=2)

    '''
    #transform y-direction
    Uc_hat_y[:]= ifft(fu,axis=1)

    plt.imshow((real(Uc_hat_y[:, -1, :])))
    plt.show()


    # Roll to x axis
    Uc_hat_x[:] = rollaxis(Uc_hat_y.reshape((N2, P2, N2, int(N1/2))), 1).reshape(Uc_hat_x.shape)

    #Communicate in xz plane
    commxz.Alltoall([Uc_hat_x, mpitype], [Uc_hat_xr, mpitype])

    #Transform in x-direction
    Uc_hat_x[:] = ifft(Uc_hat_xr, axis=0)


    #communicate in xy-plane
    commxy.Alltoall([Uc_hat_x, mpitype], [Uc_hat_xr, mpitype])
    #roll to z axis
    Uc_hat_z[:, :, :-1] = rollaxis(Uc_hat_xr.reshape((P1, N1, N2, int(N_half/P1))), 0, 3).reshape(
        (N1, N2, int(N_nyquist)-1))



    #transform in z-direction
    u[:]=irfft(Uc_hat_z,axis=2)
    '''
    return u


def fftn_mpi(u, fu):
    # FFT in three directions using MPI and the pencil decomposition
    Uc_hat_z[:]=rfft(u,axis=2)


    # Transform to x direction neglecting neglecting k=N/2 (Nyquist)
    Uc_hat_x[:] = rollaxis(Uc_hat_z[:,:,:-1].reshape((N1,N2,P1,int(N1/2))),2).reshape(Uc_hat_x.shape)




    # Communicate and do FFT in x-direction
    commxz.Alltoall([Uc_hat_x,mpitype],[Uc_hat_xr,mpitype])
    Uc_hat_x[:]=fft(Uc_hat_xr,axis=0)



    # Communicate and do fft in y-direction
    commxy.Alltoall([Uc_hat_x, mpitype], [Uc_hat_xr, mpitype])
    Uc_hat_y[:] = rollaxis(Uc_hat_xr.reshape((P2,N2,N2,int(N_half/P1))),1).reshape(Uc_hat_y.shape)



    fu[:]=fft(Uc_hat_y,axis=1)
    return fu


def Cross(a, b, c):
    # 3D cross product
    c[0] = fftn_mpi(a[1] * b[2] - a[2] * b[1], c[0])
    c[1] = fftn_mpi(a[2] * b[0] - a[0] * b[2], c[1])
    c[2] = fftn_mpi(a[0] * b[1] - a[1] * b[0], c[2])
    return c


def Curl(a, c):
    # 3D curl operator
    c[2] = ifftn_mpi(1j * (K[0] * a[1] - K[1] * a[0]), c[2])
    c[1] = ifftn_mpi(1j * (K[2] * a[0] - K[0] * a[2]), c[1])
    c[0] = ifftn_mpi(1j * (K[1] * a[2] - K[2] * a[1]), c[0])
    return c


def computeRHS(dU, rk):
    # Compute residual of time integral as specified in pseudo spectral Galerkin method
    if rk > 0:
        for i in range(3):
            U[i] = ifftn_mpi(U_hat[i], U[i])
    curl[:] = Curl(U_hat, curl)
    dU = Cross(U, curl, dU)
    dU *= dealias
    P_hat[:] = sum(dU * K_over_K2, 0, out=P_hat)
    dU -= P_hat * K
    dU -= nu * K2 * U_hat
    return dU


def spectrum(length,u,v,w):
    data_path = "./"

    Figs_Path = "./"
    Fig_file_name = "Ek_Spectrum"

    # -----------------------------------------------------------------
    #  COMPUTATIONS
    # -----------------------------------------------------------------
    localtime = time.asctime(time.localtime(time.time()))
    print("Computing spectrum... ", localtime)



    N = int(round((length ** (1. / 3))))
    print("N =", N)
    eps = 1e-50  # to void log(0)

    U = u
    V = v
    W = w

    amplsU = abs(fftn(U) / U.size)
    amplsV = abs(fftn(V) / V.size)
    amplsW = abs(fftn(W) / W.size)

    EK_U = amplsU ** 2
    EK_V = amplsV ** 2
    EK_W = amplsW ** 2

    EK_U = fftshift(EK_U)
    EK_V = fftshift(EK_V)
    EK_W = fftshift(EK_W)

    sign_sizex = shape(EK_U)[0]
    sign_sizey = shape(EK_U)[1]
    sign_sizez = shape(EK_U)[2]

    box_sidex = sign_sizex
    box_sidey = sign_sizey
    box_sidez = sign_sizez

    box_radius = int(ceil((sqrt((box_sidex) ** 2 + (box_sidey) ** 2 + (box_sidez) ** 2)) / 2.) + 1)

    centerx = int(box_sidex / 2)
    centery = int(box_sidey / 2)
    centerz = int(box_sidez / 2)

    print("box sidex     =", box_sidex)
    print("box sidey     =", box_sidey)
    print("box sidez     =", box_sidez)
    print("sphere radius =", box_radius)
    print("centerbox     =", centerx)
    print("centerboy     =", centery)
    print("centerboz     =", centerz, "\n")

    EK_U_avsphr = zeros(box_radius, ) + eps  ## size of the radius
    EK_V_avsphr = zeros(box_radius, ) + eps  ## size of the radius
    EK_W_avsphr = zeros(box_radius, ) + eps  ## size of the radius

    for i in range(box_sidex):
        for j in range(box_sidey):
            for k in range(box_sidez):
                wn = int(round(sqrt((i - centerx) ** 2 + (j - centery) ** 2 + (k - centerz) ** 2)))
                EK_U_avsphr[wn] = EK_U_avsphr[wn] + EK_U[i, j, k]
                EK_V_avsphr[wn] = EK_V_avsphr[wn] + EK_V[i, j, k]
                EK_W_avsphr[wn] = EK_W_avsphr[wn] + EK_W[i, j, k]

    EK_avsphr = 0.5 * (EK_U_avsphr + EK_V_avsphr + EK_W_avsphr)

    fig2 = plt.figure()
    plt.title("Kinetic Energy Spectrum")
    plt.xlabel(r"k (wavenumber)")
    plt.ylabel(r"TKE of the k$^{th}$ wavenumber")

    realsize = len(rfft(U[:, 0, 0]))
    plt.loglog(arange(0, realsize), ((EK_avsphr[0:realsize])), 'k')
    plt.loglog(arange(realsize, len(EK_avsphr), 1), ((EK_avsphr[realsize:])), 'k--')
    axes = plt.gca()
    axes.set_ylim([10 ** -25, 5 ** -1])

    print("Real      Kmax    = ", realsize)
    print("Spherical Kmax    = ", len(EK_avsphr))

    TKEofmean_discrete = 0.5 * (sum(U / U.size) ** 2 + sum(W / W.size) ** 2 + sum(W / W.size) ** 2)
    TKEofmean_sphere = EK_avsphr[0]

    total_TKE_discrete = sum(0.5 * (U ** 2 + V ** 2 + W ** 2)) / (N * 1.0) ** 3
    total_TKE_sphere = sum(EK_avsphr)

    print("the KE  of the mean velocity discrete  = ", TKEofmean_discrete)
    print("the KE  of the mean velocity sphere    = ", TKEofmean_sphere)
    print("the mean KE discrete  = ", total_TKE_discrete)
    print("the mean KE sphere    = ", total_TKE_sphere)

    localtime = time.asctime(time.localtime(time.time()))
    print("Computing spectrum... ", localtime, "- END \n")

    # -----------------------------------------------------------------
    #  OUTPUT/PLOTS
    # -----------------------------------------------------------------

    dataout = zeros((box_radius, 2))
    dataout[:, 0] = arange(0, len(dataout))
    dataout[:, 1] = EK_avsphr[0:len(dataout)]

    savetxt(Figs_Path + Fig_file_name + '.dat', dataout)
    fig.savefig(Figs_Path + Fig_file_name + '.pdf')
    plt.show()

def reshapeGathered(u_gathered,N,N1,N2,P1,P2,num_processes):
    if num_processes == 1:
        u_reshaped = reshape(u_gathered, (3, N, N, N))
    else:
        store_vector = empty(shape=(P1,P2)).tolist()
        #row_Vectors = empty(shape=(P1)).tolist()
        counter=0
        for j in range(P2):
            for i in range(P1):

                store_vector[i][j]=u_gathered[counter]
                counter +=1

        first_concat = concatenate(store_vector,axis=2)
        second_concat = concatenate(first_concat,axis=2)
        u_reshaped = second_concat
    # new vector = concatenate((vec1,vec2,vec3,---),axis=..)
    # u_gathered has shape (N_process,3,N/N1,N/N2,N)
    # P1 = 2
    # P2 = num_process/P1
    # N1 = N/P1 = N/2 = 8
    # N2 = N/P2 = 2*N/num_process = 8
    # we want shape (3,N,N,N)
    # need to loop over number of processes and concatenate in x-direction
    # P1 times, and in y-direction P2
    return u_reshaped


# initial condition and transformation to Fourier space
U[0] = sin(X[0]) * cos(X[1]) * cos(X[2])
U[1] = -cos(X[0]) * sin(X[1]) * cos(X[2])
U[2] = 0

#print('plot before transform')
#plt.imshow(K2[0,:,:])
#plt.imshow(U[0][:,:,int(N/2)],cmap='jet')
#plt.show()

if __name__ == '__main__':


    for i in range(3):
        U_hat[i] = fftn_mpi(U[i], U_hat[i])

    # Time integral using a Runge Kutta scheme
    t = 0.0
    tstep = 0
    mid_idx = int(N / 2)
    pbar = tqdm(total=int(T / dt))
    plotting = 'plot'
    fig = plt.figure()
    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    
    while t < T - 1e-8:
    
        t += dt;
        U_hat1[:] = U_hat0[:] = U_hat
        for rk in range(4):
            # Run RK4 temporal integral method
            dU = computeRHS(dU, rk)
            if rk < 3: U_hat[:] = U_hat0 + b[rk] * dt * dU
            U_hat1[:] += a[rk] * dt * dU
        U_hat[:] = U_hat1[:]
        for i in range(3):
            # Inverse Fourier transform after RK4 algorithm
            U[i] = ifftn_mpi(U_hat[i], U[i])
    
    
        if tstep%50==0:
            u_gathered = comm.gather(U, root=0)

            if rank==0:
                u_reshaped = reshapeGathered(u_gathered,N,N1,N2,P1,P2,num_processes)


                if plotting == 'animation':
                    im = plt.imshow(U_test[0][:,:,int(N/2)],cmap='jet', animated=True)
                    ims.append([im])
                if plotting == 'plot':
                   # print('shape of gathered U',shape(u_plot))
                   # print('shape of gathered U, concatenated',shape(U_test))
                    plt.imshow(u_reshaped[0][:,:,-1],cmap='jet')
                    plt.pause(0.05)
                    #plt.pause(0.05)
                if plotting == 'spectrum':
                    spectrum(N, U_test[0], U_test[1], U_test[2])

        tstep += 1
        pbar.update(1)
    
    k = comm.reduce(0.5 * sum(U * U) * (1. / N) ** 3)
    # if rank == 0:
    #   assert round(k - 0.124953117517, 7) == 0
    pbar.close()
    
    
    if plotting=='animation':
        if rank==0:
            ani = animation.ArtistAnimation(fig, ims, interval=2, blit=True,
                                            repeat_delay=None)
            ani.save('animationVelocity.gif', writer='imagemagick')
