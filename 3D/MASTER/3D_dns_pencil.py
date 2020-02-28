from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, fftn,fftshift,rfft,irfft
from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from numba import jit, float32 as numfloat32
try:
    from tqdm import tqdm
except ImportError:
    pass
from mpistuff.mpibase import work_arrays
work_array = work_arrays()

#TODO Make nice comments on all the functions and different parts of the script
###############################################
###############################################
# USER CHOICE ##
nu = 0.000625
Tend = 10
dt = 0.01
N_tsteps = ceil(Tend/dt)
bool_percentile = 0.10
plotting = 'noFigure'
L = 2*pi
N = int(2 ** 5)
###############################################
###############################################

N_half = int(N / 2)
N_nyquist=int(N/2+1)
P1 = 1

# Initialize MPI communication and set number of processes along each axis.
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
if num_processes > 1:
    ############
    # USER CHOICE
    # Make sure this number is smaller than nr of core available.
    P1 = 2
    ############
mpitype = MPI.DOUBLE_COMPLEX
rank = comm.Get_rank()
P2 = int(num_processes/P1)
N1 = int(N/P1)
N2 = int(N/P2)
commxz = comm.Split(rank/P1)
commxy = comm.Split(rank%P1)
xzrank = commxz.Get_rank()
xyrank = commxy.Get_rank()

# Declaration of physical mesh
x1 = slice(xzrank*N1,(xzrank+1)*N1,1)
x2 = slice(xyrank*N2,(xyrank+1)*N2,1)
X = mgrid[x1,x2,:N].astype(float32)*L/N
# Declaration of wave numbers (Fourier space)
kx = fftfreq(N, 1. / N)
kz = kx[:(N_half)].copy();
kz[-1] *= -1
k2 = slice(int(xyrank*N2),int((xyrank+1)*N2),1)
k1 = slice(int(xzrank*N1/2),int(xzrank*N1/2 + N1/2),1)
K = array(meshgrid(kx[k2],kx,kx[k1],indexing='ij'),dtype=int)

# Preallocate arrays, decomposed using a 2D-pencil approach
U = empty((3, N1, N2, N), dtype=float32)
U_hat = empty((3, N2, N, int(N_half/P1)), dtype=complex)
P = empty((N1, N2, N),dtype=float32)
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
curl = empty((3, N1, N2, N),dtype=float32)

# Precompute wave number operators and dealias matrix.
K2 = sum(K * K, 0, dtype=int)
K_over_K2 = K.astype(float32) / where(K2 == 0, 1, K2).astype(float32)
kmax_dealias = 2. / 3. * (N_half)
dealias = array(
    (abs(K[0]) < kmax_dealias) * (abs(K[1]) < kmax_dealias) * (abs(K[2]) < kmax_dealias),
    dtype=bool)

# Runge Kutta constants
a = [1. / 6., 1. / 3., 1. / 3., 1. / 6.]
b = [0.5, 0.5, 1.]
#dir = '/home/danieloh/PycharmProjects/Project_Turbulence_Modelling/animation_folder/'

def ifftn_mpi_slab(fu, u):
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


def Cross_slow(a, b, c):
    # 3D cross product
    c[0] = fftn_mpi(a[1] * b[2] - a[2] * b[1], c[0])
    c[1] = fftn_mpi(a[2] * b[0] - a[0] * b[2], c[1])
    c[2] = fftn_mpi(a[0] * b[1] - a[1] * b[0], c[2])
    return c

@jit(nopython=True)
def Cross(a,b,c):
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            for k in range(a.shape[3]):
                a0 = a[0,i,j,k]
                a1 = a[1,i,j,k]
                a2 = a[2,i,j,k]
                b0 = b[0,i,j,k]
                b1 = b[1,i,j,k]
                b2 = b[2,i,j,k]
                c[0,i,j,k]=a1*b2-a2*b1
                c[1,i,j,k]=a2*b0-a0*b2
                c[2,i,j,k]=a0*b1-a1*b0
    return c

def Cross_T(c, a, b, work_array):
    """c_k = F_k(a x b)"""
    d = work_array[(a, 1, False)]
    d = Cross(a,b,d)
    c[0] = fftn_mpi(d[0], c[0])
    c[1] = fftn_mpi(d[1], c[1])
    c[2] = fftn_mpi(d[2], c[2])
    return c


def Curl(a, c):
    # 3D curl operator
    c[2] = ifftn_mpi(1j * (K[0] * a[1] - K[1] * a[0]), c[2])
    c[1] = ifftn_mpi(1j * (K[2] * a[0] - K[0] * a[2]), c[1])
    c[0] = ifftn_mpi(1j * (K[1] * a[2] - K[2] * a[1]), c[0])
    return c

@jit(nopython=True,fastmath=True)
def cross2a(c, a, b):
    """ c = 1j*(a x b)"""
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            for k in range(a.shape[3]):
                a0 = a[0, i, j, k]
                a1 = a[1, i, j, k]
                a2 = a[2, i, j, k]
                b0 = b[0, i, j, k]
                b1 = b[1, i, j, k]
                b2 = b[2, i, j, k]
                c[0, i, j, k] = -(a1*b2.imag - a2*b1.imag) + 1j*(a1*b2.real - a2*b1.real)
                c[1, i, j, k] = -(a2*b0.imag - a0*b2.imag) + 1j*(a2*b0.real - a0*b2.real)
                c[2, i, j, k] = -(a0*b1.imag - a1*b0.imag) + 1j*(a0*b1.real - a1*b0.real)
    return c

def compute_curl(c, a, work, K):
    """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
    curl_hat = work[(a, 0, False)]
    curl_hat = cross2a(curl_hat, K, a)
    c[0] = ifftn_mpi(curl_hat[0], c[0])
    c[1] = ifftn_mpi(curl_hat[1], c[1])
    c[2] = ifftn_mpi(curl_hat[2], c[2])
    return c

def computeRHS(dU, rk):
    # Compute residual of time integral as specified in pseudo spectral Galerkin method
    if rk > 0:
        for i in range(3):
            U[i] = ifftn_mpi(U_hat[i], U[i])

    #curl[:] = Curl(U_hat, curl)
    curl[:] = compute_curl(curl, U_hat, work_array, K)
    dU = Cross_T(dU, U, curl, work_array)
    #dU = Cross_slow(U, curl, dU)


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


def reshape_loop(P1,P2,store_vector,u_gathered):
    #TODO Optimize using Cython
    counter = 0
    for j in range(shape(store_vector)[1]):
        for i in range(shape(store_vector)[0]):
            store_vector[i][j] = u_gathered[counter]
            counter = counter+1
    return store_vector


def reshapeGathered(u_gathered,N,N1,N2,P1,P2,num_processes,method):
    if num_processes == 1:
        u_reshaped = reshape(u_gathered, (3, N, N, N))
    else:
        if method=='concatenate':
            store_vector = empty(shape=(P1, P2),dtype=float32).tolist()
            store_vector = reshape_loop(P1,P2,store_vector,u_gathered)

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

if __name__ == '__main__':

    for i in range(3):
        U_hat[i] = fftn_mpi(U[i], U_hat[i])

    t = 0.0
    tstep = 0
    plot_step = N_tsteps*bool_percentile
    fig = plt.figure()
    ims = []
    mid_idx = int(N / 2)
    try:
        pbar = tqdm(total=int(Tend / dt))
    except:
        pass

    '''
    if rank==0:
        progressfile = open('progressfile.txt','w+')
        progressfile.truncate(0)
    '''

    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame

    
    while t < Tend - 1e-8:
        # Time integral using a Runge Kutta scheme
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
    
    
        if tstep%plot_step==0:
            u_gathered = comm.gather(U, root=0)
            if rank==0:
                u_reshaped = reshapeGathered(u_gathered,N,N1,N2,P1,P2,num_processes,method='concatenate')
                if plotting == 'animation':
                    im = plt.imshow(u_reshaped[0][:,:,-1],cmap='jet', animated=True)
                    ims.append([im])
                if plotting == 'plot':
                    plt.imshow(u_reshaped[0][:,:,-1],cmap='jet')
                    plt.pause(0.05)
                    #plt.pause(0.05)
                if plotting == 'spectrum':
                    spectrum(N, u_reshaped[0], u_reshaped[1], u_reshaped[2])
                if plotting == 'savefig':
                    plt.imshow(u_reshaped[0][:, :, -1], cmap='jet')
                    plt.savefig('images/turb_t_'+str(int(tstep)))
                if plotting=='noFigure':
                    print('next iteration')
                if plotting =='saveNumpy':
                    save('vel_files/velocity_'+str(tstep)+'.npy',u_reshaped)

        tstep += 1
        #if rank ==0:
            #progressfile.write("tstep= %d\r\n" % (tstep),flush=True)
            #print('tstep= %d\r\n'%(tstep),flush=True)
        try:
            pbar.update(1)
        except:
            pass

    if plotting=='animation':
        if rank==0:
            ani = animation.ArtistAnimation(fig, ims, interval=2, blit=True,
                                            repeat_delay=None)
            ani.save('animationVelocity.gif', writer='imagemagick')
