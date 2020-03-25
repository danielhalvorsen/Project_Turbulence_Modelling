from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, fftn,fftshift,rfft,irfft
from mpi4py import MPI
import matplotlib.pyplot as plt
from sys import getsizeof
import matplotlib.animation as animation
import time
from Particle_mpi import Interpolator,Euler,periodicBC,plotScatter,particle_IC,trajectory
from numba import jit, float32 as numfloat32
try:
    from tqdm import tqdm
except ImportError:
    pass
from mpistuff.mpibase import work_arrays
work_array = work_arrays()

#TODO Make nice comments on all the functions and different parts of the script
#############################################################################################################################################

## USER CHOICE DNS ##

#############################################################################################################################################
nu = 0.000625
Tend = 10
dt = 0.01
N_tsteps = int(ceil(Tend/dt))
bool_percentile = 0.02
plotting = 'saveNumpy'
IC = 'isotropic2'
L = 2*pi
eta = 2*pi*((1/nu)**(-3/4))
N = int(2 ** 9)
N_three = N**3
force = 0.00078
kf = 8           #Highest wave number that is forced.
Re_lam = 128
k0 = 1.7 #Kolmogorov constant
kd = 1/eta #Dissipative length scale, inverse of kolmogorov length scale \eta.
target = Re_lam*(nu*kd)**2/(sqrt(20./3.)) # Energy of flow with given parameters of Re_lam, kd and nu.

#############################################################################################################################################

## Hard constants for mesh and MPI communication

#############################################################################################################################################

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
    P1 = 32
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
randomNr = random.rand(N1,N2,N)

# Declaration of wave numbers (Fourier space)
kx = fftfreq(N, 1. / N)
kz = kx[:(N_half)].copy();
kz[-1] *= -1
k2 = slice(int(xyrank*N2),int((xyrank+1)*N2),1)
k1 = slice(int(xzrank*N1/2),int(xzrank*N1/2 + N1/2),1)
K = array(meshgrid(kx[k2],kx,kx[k1],indexing='ij'),dtype=int)

kx_single = fftfreq(N, 1. / N)
K_single = array(meshgrid(kx,kx, kz, indexing='ij'), dtype=int)
K2_single = sum(K_single * K_single, 0, dtype=int)
eps = 1e-6
kinBand = 1

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
k2_mask = where(K2 <= kf**2, 1, 0)

#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################

## Define Particle parameters ##

#############################################################################################################################################
multipleNp = 400
Np = num_processes*multipleNp
ldx = L / N
#particle_X = zeros((N_tsteps+1,3,Np))
par_Pos_init = particle_IC(Np,L)
particle_save_array=None
if rank==0:
    particle_save_array = array([par_Pos_init.copy()])
print('Shape of Par_Pos_init: '+str(shape(par_Pos_init)),flush=True)
#particle_X[0]=par_Pos_init
#print('Shape of particle_X: '+str(shape(particle_X)),flush=True)

split_coordinates = array_split(par_Pos_init,num_processes,axis=1)
print('Shape of split_coordinates: '+str(shape(split_coordinates)),flush=True)
#particleData = empty((N_tsteps+1,3,))
particleData_old = comm.scatter(split_coordinates,root=0)
print('Shape of particleData_old: '+str(shape(particleData_old)),flush=True)


fig_particle = plt.figure()
ax_particle = fig_particle.add_subplot(111, projection='3d')
pointSize = 0.1
pointcolor1 = 'r'
pointcolor2 = 'm'
rgbaTuple = (167/255, 201/255, 235/255)

#TODO wont need Tmax unless we collect new velocity field every time step.
h  = dt
t0 = 0


# Runge Kutta constants
a = [1. / 6., 1. / 3., 1. / 3., 1. / 6.]
b = [0.5, 0.5, 1.]

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
    #TODO fix buffer error for np=1
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
    return u_reshaped


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

@jit(nopython=True)
def dissipationLoop(wave_numbers,nu,tke):
    sum = 0
    wavesquare = [x**2 for x in wave_numbers]
    for k in range(shape(wavesquare)[0]):
        sum += (wavesquare[k]*tke[k])
    return real(2*nu*sum)

@jit(nopython=True)
def BandEnergy(tke,kf):
    sum = 0
    for k in range(kf):
        sum += (tke[k])
    return sum


def integralEnergy(comm,arg):
    #TODO make this function work in parallel?
    result = ((sum(abs(arg[...]) ** 2)))
    return comm.allreduce(result/N_three)

@jit(nopython=True)
def kloop(nx,ny,nz,tke_spectrum,tkeh):
    for kx in range(-nx//2, nx//2-1):
        for ky in range(-ny//2, ny//2-1):
            for kz in range(-nz//2, nz//2-1):
                rk = sqrt(kx**2 + ky**2 + kz**2)
                k = int(round(rk))
                tke_spectrum[k] += tkeh[kx, ky, kz]
    return tke_spectrum

def movingaverage(interval, window_size):
    window = ones(int(window_size)) / float(window_size)
    return convolve(interval, window, 'same')

def compute_tke_spectrum(u, v, w, smooth):
    """
    Given a velocity field u, v, w, this function computes the kinetic energy
    spectrum of that velocity field in spectral space. This procedure consists of the
    following steps:
    1. Compute the spectral representation of u, v, and w using a fast Fourier transform.
    This returns uf, vf, and wf (the f stands for Fourier)
    2. Compute the point-wise kinetic energy Ef (kx, ky, kz) = 1/2 * (uf, vf, wf)* conjugate(uf, vf, wf)
    3. For every wave number triplet (kx, ky, kz) we have a corresponding spectral kinetic energy
    Ef(kx, ky, kz). To extract a one dimensional spectrum, E(k), we integrate Ef(kx,ky,kz) over
    the surface of a sphere of radius k = sqrt(kx^2 + ky^2 + kz^2). In other words
    E(k) = sum( E(kx,ky,kz), for all (kx,ky,kz) such that k = sqrt(kx^2 + ky^2 + kz^2) ).
    Parameters:
    -----------
    u: 3D array
      The x-velocity component.
    v: 3D array
      The y-velocity component.
    w: 3D array
      The z-velocity component.
    lx: float
      The domain size in the x-direction.
    ly: float
      The domain size in the y-direction.
    lz: float
      The domain size in the z-direction.
    smooth: boolean
      A boolean to smooth the computed spectrum for nice visualization.
    """
    lx,ly,lz = L,L,L
    nx,ny,nz = N,N,N

    nt = nx * ny * nz
    n = nx  # int(np.round(np.power(nt,1.0/3.0)))

    uh = fftn(u) / nt
    vh = fftn(v) / nt
    wh = fftn(w) / nt

    tkeh = 0.5 * (uh * conj(uh) + vh * conj(vh) + wh * conj(wh)).real

    k0 = 2.0 * pi / lx
    knorm = (k0 + k0 + k0) / 3.0
    kmax = n / 2

    # dk = (knorm - kmax)/n
    # wn = knorm + 0.5 * dk + arange(0, nmodes) * dk

    wave_numbers = knorm * arange(0, n)

    tke_spectrum = zeros(len(wave_numbers))
    tke_spectrum = kloop(nx,ny,nz,tke_spectrum,tkeh)
    tke_spectrum = tke_spectrum / knorm

    if smooth:
        tkespecsmooth = movingaverage(tke_spectrum, 5)  # smooth the spectrum
        tkespecsmooth[0:4] = tke_spectrum[0:4]  # get the first 4 values from the original data
        tke_spectrum = tkespecsmooth

    knyquist = knorm * min(nx, ny, nz) / 2

    return knyquist, wave_numbers, tke_spectrum

def initialize2(rank,K,dealias,K2,N,U_hat):
    random.seed(rank)
    k = sqrt(K2)
    k = where(k == 0, 1, k)
    kk = K2.copy()
    kk = where(kk == 0, 1, kk)
    k1, k2, k3 = K[0], K[1], K[2]
    ksq = sqrt(k1 ** 2 + k2 ** 2)
    ksq = where(ksq == 0, 1, ksq)

    if (N == (2 ** 5)):
        C = 260
        a = 2.5
    elif (N == (2 ** 6)):
        C = 2600
        a = 3.5
    elif (N == (2 ** 7)):
        C = 5000
        a = 3.5
    elif(N==(2**8)):
        C=2600
        a=3.5
    elif (N == (2 ** 9)):
        C = 10000
        a = 9.5
    Ek = (C*abs(k)*2*N_three/((2*pi)**3))*exp((-abs(kk))/(a**2))
    # theta1, theta2, phi, alpha and beta from [1]
    theta1, theta2, phi = random.sample(U_hat.shape) * 2j * pi
    alpha = sqrt(Ek / 4. / pi / kk) * exp(1j * theta1) * cos(phi)
    beta = sqrt(Ek / 4. / pi / kk) * exp(1j * theta2) * sin(phi)
    U_hat[0] = (alpha * k * k2 + beta * k1 * k3) / (k * ksq)
    U_hat[1] = (beta * k2 * k3 - alpha * k * k1) / (k * ksq)
    U_hat[2] = beta * ksq / k

    # project to zero divergence
    U_hat[:] -= (K[0] * U_hat[0] + K[1] * U_hat[1] + K[2] * U_hat[2]) * K_over_K2

    '''
    energy = 0.5 * integralEnergy(comm,U_hat)
    U_hat *= sqrt(target / energy)
    energy= 0.5 * integralEnergy(comm,U_hat)
    '''
    return U_hat

def initialize(rank,K,dealias,K2,N,U_hat):

    # Create mask with ones where |k| < Kf2 and zeros elsewhere
    #kf = 5
    #k2_mask = where(K2 <= kf**2, 1, 0)
    #k2_mask = dealias
    random.seed(rank)
    k = sqrt(K2)
    k = where(k == 0, 1, k)
    kk = K2.copy()
    kk = where(kk == 0, 1, kk)
    k1, k2, k3 = K[0], K[1], K[2]
    ksq = sqrt(k1**2+k2**2)
    ksq = where(ksq == 0, 1, ksq)

    E0 = sqrt(9./11./kf*K2/((kf)**2))*k2_mask
    E1 = sqrt(9./11./kf*(k/kf)**(-5./3.))*(1-k2_mask)
    Ek = E0 + E1
    # theta1, theta2, phi, alpha and beta from [1]
    theta1, theta2, phi = random.sample(U_hat.shape)*2j*pi
    alpha = sqrt(Ek/4./pi/kk)*exp(1j*theta1)*cos(phi)
    beta = sqrt(Ek/4./pi/kk)*exp(1j*theta2)*sin(phi)
    U_hat[0] = (alpha*k*k2 + beta*k1*k3)/(k*ksq)
    U_hat[1] = (beta*k2*k3 - alpha*k*k1)/(k*ksq)
    U_hat[2] = beta*ksq/k




    # project to zero divergence
    U_hat[:] -= (K[0]*U_hat[0]+K[1]*U_hat[1]+K[2]*U_hat[2])*K_over_K2


    energy = 0.5 * integralEnergy(comm,U_hat)
    U_hat *= sqrt(target / energy)
    energy= 0.5 * integralEnergy(comm,U_hat)

    return U_hat

def computeRHS(dU, rk):
    # Compute residual of time integral as specified in pseudo spectral Galerkin method
    # TODO add forcing term here?
    if rk > 0:
        for i in range(3):
            U[i] = ifftn_mpi(U_hat[i], U[i])

    curl[:] = compute_curl(curl, U_hat, work_array, K)
    dU = Cross_T(dU, U, curl, work_array)

    dU *= dealias
    P_hat[:] = sum(dU * K_over_K2, 0, out=P_hat)
    dU -= P_hat * K
    dU -= nu * K2 * U_hat


    #TODO activate this source term
    dU += (force*U_hat*k2_mask/(2*kinBand))
    return dU
def dynamicPostProcess(tstep,plot_step):
    if tstep % plot_step == 0:
        energy_new = integralEnergy(comm, U_hat)
        if rank == 0:
            print('\nEnergy in system:    ' + str(energy_new), flush=True)
        u_gathered = comm.gather(U, root=0)
        if rank == 0:
            u_reshaped = reshapeGathered(u_gathered, N, N1, N2, P1, P2, num_processes, method='concatenate')
            nyquist, k, tke = compute_tke_spectrum(u_reshaped[0], u_reshaped[1], u_reshaped[2], True)
            eps = dissipationLoop(k, nu, tke)
            print('\nDissipation:   ' + str(eps), flush=True)
            kinBand = BandEnergy(tke, kf)
            kinTotal = BandEnergy(tke, k[-1])
            if plotting == 'animation':
                im = plt.imshow(u_reshaped[0][:, :, -1], cmap='jet', animated=True)
                ims.append([im])
            if plotting == 'plot':
                plt.imshow(u_reshaped[0][:, :, -1])
                plt.pause(0.05)
                # plt.pause(0.05)
            if plotting == 'EnergyTotal':
                energyarrayKin.append(kinTotal)
                plt.plot(t_array[0:len(energyarrayKin)] * (plot_step), energyarrayKin, 'r--')
                plt.xlabel('Time, (s)')
                plt.ylabel('$E_{kin}$')
                plt.pause(0.05)
            if plotting == 'EnergyKf':
                energyarrayKf.append(tke[kf])
                plt.plot(t_array[0:len(energyarrayKf)] * (plot_step), energyarrayKf, 'r--')
                plt.xlabel('Time, (s)')
                plt.ylabel('E(kf)')
                plt.pause(0.05)
            if plotting == 'spectrum':
                # TODO plot the spectrum with k\eta such that the last place y-axis crosses, is at kolmogorov scale.
                plt.loglog(k[1:N_half], tke[1:N_half] * (eps ** (-2 / 3)), 'g.', 'markerSize=2')
                plt.loglog(k[1:N_half], (k[1:N_half] ** (-5 / 3)) * (eps ** (-2 / 3)), 'r--')
                plt.yscale('log')
                plt.ylim(ymin=(1e-18), ymax=1e3)
                plt.xlabel('Wave number, $k$')
                plt.ylabel('Turbulent kinetic energy, $E(k)$')
                plt.legend(['$E(k)$,  t= %.2f' % (tstep / 100), r'$\epsilon^{-2/3}k^{-5/3}$'], loc='lower left')
                plt.pause(0.05)
                plt.close()
            if plotting == 'savefig':
                plt.imshow(u_reshaped[0][:, :, -1], cmap='jet')
                plt.savefig('images/turb_t_' + str(int(tstep)))
            if plotting == 'noFigure':
                print('next iteration')
            if plotting == 'saveNumpy':
                save('vel_files_iso/velocity_' + str(tstep) + '.npy', u_reshaped)
def mpiPrintIteration(tstep):
    if rank == 0:
        # progressfile.write("tstep= %d\r\n" % (tstep),flush=True)
        print('tstep= %d\r\n' % (tstep), flush=True)
def oldEnergyUpdate():
    '''
    energy_new = integralEnergy(comm, U_hat)
    energy_lower = integralEnergy(comm, U_hat * k2_mask)
    energy_upper = energy_new - energy_lower
    alpha2 = (target_energy - energy_upper) / energy_lower
    alpha = sqrt(alpha2)

    U_hat *= (alpha * k2_mask + (1 - k2_mask))
    energy_new = integralEnergy(comm, U_hat)
    if rank == 0:
        print(energy_new)
    assert sqrt((energy_new - target_energy) ** 2) < 1e-7, sqrt((energy_new - target) ** 2)
    '''
    return 0
def compueParticle(U,num_processes,particleData_old):
    #todo implement multiprocess for trajectory
    u_gathered = comm.gather(U, root=0)
    if rank == 0:
        u_reshaped = reshapeGathered(u_gathered, N, N1, N2, P1, P2, num_processes, 'concatenate')
    else:
        u_reshaped = None
    u_reshaped = comm.bcast(u_reshaped, root=0)
    f = Interpolator(u_reshaped)
    particleData_new = trajectory(t0, Tend, h, f, Euler,True,L,ldx,particleData_old)

    return particleData_new


def saveParticleCoord(particleData_new,particle_save_array,tstep):
    particleData_new_gathered = comm.gather(particleData_new, root=0)
    if rank == 0:
        particleData_new_reshaped = array([concatenate(particleData_new_gathered, axis=1)])
        particle_save_array= concatenate((particle_save_array,particleData_new_reshaped),axis=0)
        if (tstep%50==0):
            save('particleCoord.npy', particle_save_array)
            print('save at tstep: '+str(tstep),flush=True)
        return particle_save_array

if __name__ == '__main__':
    # initial condition and transformation to Fourier space
    if IC == 'isotropic2':
        U_hat = initialize2(rank, K, dealias, K2,N,U_hat)
        for i in range(3):
            U[i] = ifftn_mpi(U_hat[i], U[i])
    if IC == 'TG':
        U[0] = sin(X[0]) * cos(X[1]) * cos(X[2])
        U[1] = -cos(X[0]) * sin(X[1]) * cos(X[2])
        U[2] = 0
        for i in range(3):
            U_hat[i] = fftn_mpi(U[i], U_hat[i])
    target_energy = integralEnergy(comm,U_hat)
    t = 0.0
    tstep = 0
    t_array = arange(0,Tend,dt)
    energyarrayKf = []
    energyarrayKin = []
    plot_step = N_tsteps*bool_percentile
    #fig = plt.figure()
    ims = []
    mid_idx = int(N / 2)
    try:
        pbar = tqdm(total=int(Tend / dt))
    except:
        pass

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



        dynamicPostProcess(tstep,plot_step)
        particleData_new = compueParticle(U,num_processes,particleData_old)
        particleData_old = particleData_new
        particle_save_array = saveParticleCoord(particleData_new, particle_save_array, tstep)


        tstep += 1
        mpiPrintIteration(tstep)


        try:
            pbar.update(1)
        except:
            pass
    # Add extra save to account for last timestep
    if rank ==0:
        save('particleCoord.npy', particle_save_array)
        print('saved last timestep',flush=True)