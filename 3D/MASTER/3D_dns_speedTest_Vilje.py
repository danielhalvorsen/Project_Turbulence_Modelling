from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, fftn,fftshift,rfft,irfft
from mpi4py import MPI
import time
from mpistuff.mpibase import work_arrays
work_array = work_arrays()

#TODO Make nice comments on all the functions and different parts of the script
#############################################################################################################################################

## USER CHOICE DNS ##

#############################################################################################################################################
nu = 0.000625
Tend = 1
dt = 0.01
N_tsteps = int(ceil(Tend/dt))
IC = 'isotropic2'
L = 2*pi
eta = 2*pi*((1/nu)**(-3/4))
N = int(2 ** 6)
kf = 8
N_three = (N**3)
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
randomNr = random.rand(N1,N2,N)

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
k2_mask = where(K2 <= kf**2, 1, 0)


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

def Cross(a, b, c):
    c[0] = fftn_mpi(a[1]*b[2]-a[2]*b[1], c[0])
    c[1] = fftn_mpi(a[2]*b[0]-a[0]*b[2], c[1])
    c[2] = fftn_mpi(a[0]*b[1]-a[1]*b[0], c[2])
    return c
#@profile
def Curl(a, c):
    c[2] = ifftn_mpi(1j*(K[0]*a[1]-K[1]*a[0]), c[2])
    c[1] = ifftn_mpi(1j*(K[2]*a[0]-K[0]*a[2]), c[1])
    c[0] = ifftn_mpi(1j*(K[1]*a[2]-K[2]*a[1]), c[0])
    return c


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



def computeRHS(dU, rk):
    # Compute residual of time integral as specified in pseudo spectral Galerkin method
    # TODO add forcing term here?
    if rk > 0:
        for i in range(3):
            U[i] = ifftn_mpi(U_hat[i], U[i])

    curl[:] = Curl(U_hat, curl)
    dU = Cross(U, curl, dU)
    dU *= dealias
    P_hat[:] = sum(dU * K_over_K2, 0, out=P_hat)
    dU -= P_hat * K
    dU -= nu * K2 * U_hat

    #dU += (force*U_hat*k2_mask/(2*kinBand))
    return dU



def mpiPrintIteration(tstep):
    if rank == 0:
        # progressfile.write("tstep= %d\r\n" % (tstep),flush=True)
        print('tstep= %d\r\n' % (tstep), flush=True)



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


    t = 0.0
    tstep = 0
    speedList = []

    while t < Tend - 1e-8:
        # Time integral using a Runge Kutta scheme
        t += dt;
        U_hat1[:] = U_hat0[:] = U_hat

        for rk in range(4):
            # Run RK4 temporal integral method

            if rank==0:
                start = time.time()
            dU = computeRHS(dU, rk)
            if rank==0:
                end = time.time()
                speed=(end-start)
                speedList.append(speed)
            if rk < 3: U_hat[:] = U_hat0 + b[rk] * dt * dU
            U_hat1[:] += a[rk] * dt * dU

        U_hat[:] = U_hat1[:]

        tstep += 1
        mpiPrintIteration(tstep)
        if rank==0:
            save('./speed_files/speedTest_t100_np'+str(num_processes)+'.npy',speedList)
