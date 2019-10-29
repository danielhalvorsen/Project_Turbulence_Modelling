from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2
from mpi4py import MPI
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

# U is set to dtype float32
# Reynoldsnumber determined by nu Re = 1600, nu = 1/1600
nu = 0.000625
# nu = 0.00000625
T = 20
dt = 0.1
N = int(2 ** 7)
N_half = int(N / 2 + 1)
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
Np = int(N / num_processes)
X = mgrid[rank * Np:(rank + 1) * Np, :N, :N].astype(float) * 2 * pi / N
# using np.empty() does not create a zero() list!
U = empty((3, Np, N, N), dtype=float32)
U_hat = empty((3, N, Np, N_half), dtype=complex)
P = empty((Np, N, N))
P_hat = empty((N, Np, N_half), dtype=complex)
U_hat0 = empty((3, N, Np, N_half), dtype=complex)
U_hat1 = empty((3, N, Np, N_half), dtype=complex)
dU = empty((3, N, Np, N_half), dtype=complex)
Uc_hat = empty((N, Np, N_half), dtype=complex)
Uc_hatT = empty((Np, N, N_half), dtype=complex)
U_mpi = empty((num_processes, Np, Np, N_half), dtype=complex)
curl = empty((3, Np, N, N))
animate_U_x = empty((int(T / dt), Np, N, N), dtype=float32)
save_animation = True
kx = fftfreq(N, 1. / N)
kz = kx[:(N_half)].copy();
kz[-1] *= -1
K = array(meshgrid(kx, kx[rank * Np:(rank + 1) * Np], kz, indexing="ij"), dtype=int)
K2 = sum(K * K, 0, dtype=int)
K_over_K2 = K.astype(float) / where(K2 == 0, 1, K2).astype(float)
kmax_dealias = 2. / 3. * (N_half)
dealias = array(
    (abs(K[0]) < kmax_dealias) * (abs(K[1]) < kmax_dealias) * (abs(K[2]) < kmax_dealias),
    dtype=bool)

a = [1. / 6., 1. / 3., 1. / 3., 1. / 6.]
b = [0.5, 0.5, 1.]


def ifftn_mpi(fu, u):
    # Inverse Fourier transform
    Uc_hat[:] = ifft(fu, axis=0)
    comm.Alltoall([Uc_hat, MPI.DOUBLE_COMPLEX], [U_mpi, MPI.DOUBLE_COMPLEX])
    Uc_hatT[:] = rollaxis(U_mpi, 1).reshape(Uc_hatT.shape)
    u[:] = irfft2(Uc_hatT, axes=(1, 2))
    return u


def fftn_mpi(u, fu):
    # Forward Fourier transform
    Uc_hatT[:] = rfft2(u, axes=(1, 2))
    U_mpi[:] = rollaxis(Uc_hatT.reshape(Np, num_processes, Np, N_half), 1)
    comm.Alltoall([U_mpi, MPI.DOUBLE_COMPLEX], [fu, MPI.DOUBLE_COMPLEX])
    fu[:] = fft(fu, axis=0)
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


# initial condition and transformation to Fourier space
U[0] = sin(X[0]) * cos(X[1]) * cos(X[2])
U[1] = -cos(X[0]) * sin(X[1]) * cos(X[2])
U[2] = 0
for i in range(3):
    U_hat[i] = fftn_mpi(U[i], U_hat[i])

# Time integral using a Runge Kutta scheme
t = 0.0
tstep = 0
save_nr = 1
mid_idx = int(N / 2)
pbar = tqdm(total=int(T / dt))
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
    # if save_animation == True and tstep % save_nr == 0:
    # Save the animation every "save_nr" time step
    animate_U_x[tstep] = U[0].copy()
    tstep += 1
    pbar.update(1)

k = comm.reduce(0.5 * sum(U * U) * (1. / N) ** 3)
# if rank == 0:
#   assert round(k - 0.124953117517, 7) == 0
pbar.close()

# Gather the scattered data and store into two variables, X and U.
# Root is rank of receiving process (core 1)
U_gathered = comm.gather(U, root=0)
X_gathered = comm.gather(X, root=0)

animate_U_x_T = animate_U_x.transpose((0, 3, 2, 1))
animate_save_T = [animate_U_x_T[i][int(N / 2)] for i in range(len(animate_U_x_T))]

with open('U' + '.pkl', 'wb') as f:
    pickle.dump([U_gathered], f)

with open('X.pkl', 'wb') as g:
    pickle.dump([X_gathered], g)

if save_animation == True:
    #   animate_U_x_gather = comm.gather(animate_U_x, root=0)
    with open('animate_U_x_' + str(rank) + '.pkl', 'wb') as h:
        pickle.dump([animate_save_T], h)
