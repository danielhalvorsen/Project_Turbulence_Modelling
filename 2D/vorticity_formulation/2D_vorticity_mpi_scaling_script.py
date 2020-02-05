# solve 2-D incompressible NS equations using spectral method

import matplotlib.pyplot as plt
from numpy import *
from numpy.random import seed, uniform
from numpy import max as npmax
from numpy.fft import fftfreq, fft, ifft, fft2, ifft2, fftshift, ifftshift
from mpi4py import MPI
import matplotlib.animation as animation

try:
    from tqdm import tqdm
except ImportError:
    pass

# parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
# sys.path.append(parent)

# parameters
tend = 100
dt = 1e-1
Nstep = int(ceil(tend / dt))
N = Nx = Ny = 16;  # grid size
t = 0
nu = 5e-3 # viscosity
ICchoice = 'omega4'


# ------------MPI setup---------
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
print('number of processes = ',num_processes)
rank = comm.Get_rank()
Np = int(N / num_processes)
# slab decomposition, split arrays in x direction in physical space, in ky direction in
# Fourier space
Uc_hat = empty((N, Np), dtype=complex)
Uc_hatT = empty((Np, N), dtype=complex)
U_mpi = empty((num_processes, Np, Np), dtype=complex)

a = [1. / 6., 1. / 3., 1. / 3., 1. / 6.]
b = [0.5, 0.5, 1.]



def ifftn_mpi(fu, u):
    Uc_hat[:] = ifftshift(ifft(fftshift(fu), axis=0))
    comm.Alltoall([Uc_hat, MPI.DOUBLE_COMPLEX], [U_mpi, MPI.DOUBLE_COMPLEX])
    Uc_hatT[:] = rollaxis(U_mpi, 1).reshape(Uc_hatT.shape)
    u[:] = ifftshift(ifft(fftshift(Uc_hatT), axis=1))
    return u


# FFT
def fftn_mpi(u, fu):
    Uc_hatT[:] = fftshift(fft(ifftshift(u), axis=1))
    U_mpi[:] = rollaxis(Uc_hatT.reshape(Np, num_processes, Np), 1)
    comm.Alltoall([U_mpi, MPI.DOUBLE_COMPLEX], [fu, MPI.DOUBLE_COMPLEX])
    fu[:] = fftshift(fft(ifftshift(fu), axis=0))
    return fu






# ----Initialize Velocity in Fourier space-----------
def IC_condition(Nx, Np, u, v, u_hat, v_hat, ICchoice, omega, omega_hat, X, Y):
    if ICchoice == 'randomVel':
        u = random.rand(Np, Ny)
        v = random.rand(Np, Ny)
      #  u = u / npmax(u)
       # v = v / npmax(v)
        u_hat = fftn_mpi(u, u_hat)
        v_hat = fftn_mpi(v, v_hat)
        omega_hat = 1j * (Kx * v_hat - Ky * u_hat);
    if ICchoice == 'Vel1':
        if rank == 0:
            # u =
            ## v =
            # u = u/npmax(u)
            # v = v/npmax(v)
            # u_hat = fftn_mpi(u, u_hat)
            # v_hat = fftn_mpi(v, v_hat)
            u_hat[2, 2] = 5 + 10j
            v_hat[2, 2] = 5 + 10j
            u_hat[5, 2] = 5 + 10j
            v_hat[5, 2] = 5 + 10j
            u_hat[2, 3] = 5 + 10j
            v_hat[2, 3] = 5 + 10j
        u = ifftn_mpi(u_hat, u)
        v = ifftn_mpi(v_hat, v)
        u = u / npmax(u)
        v = v / npmax(v)
        u_hat = fftn_mpi(u, u_hat)
        v_hat = fftn_mpi(v, v_hat)
        omega_hat = 1j * (Kx * v_hat - Ky * u_hat);
    if ICchoice == 'omegahat1':
        if rank == 0:
            random.seed(1969)
            omega_hat[0, 4] = random.uniform() + 1j * random.uniform()
            omega_hat[1, 1] = random.uniform() + 1j * random.uniform()
            omega_hat[3, 0] = random.uniform() + 1j * random.uniform()
            omega_hat[2, 3] = random.uniform() + 1j * random.uniform()
            omega_hat[5, 3] = random.uniform() + 1j * random.uniform()

        omega = abs(ifftn_mpi(omega_hat, omega))
        omega = omega / npmax(omega)
        omega_hat = fftn_mpi(omega, omega_hat)
    if ICchoice == 'omega1':
        omega = sin(X)*cos(Y)
        omega_hat = fftn_mpi(omega, omega_hat)
    if ICchoice == 'omega3':
        H = exp(-((2*X - pi + pi / 5) ** 2 + (4*Y - pi + pi / 5) ** 2) / 0.3) - exp(
            -((2*X - pi - pi / 5) ** 2 + (3*Y - pi + pi / 5) ** 2) / 0.2) + exp(
            -((3*X - pi - pi / 5) ** 2 + (2*Y - pi - pi / 5) ** 2) / 0.4)+exp(-((2*X - pi + pi / 5)**2 + (Y - pi + pi / 5) ** 2) / 0.3) - exp(
            -((X - pi - pi / 5)**2 + (Y - pi + pi / 5) ** 2) /0.2) + exp(-((X - pi - pi / 5)**2 + (3*Y - pi - pi / 5)**2)/0.4)+\
            exp(-((X - pi + pi / 5)**2 + (Y - pi + pi / 5) ** 2) / 0.3) + exp(
            -((X - pi - pi / 5)**2 + (3*Y - pi + pi / 5) ** 2) /0.3) - exp(-((X - pi - pi / 5)**2 + (Y - pi - pi / 5)**2)/0.4);
        epsilon = 0.4;
        Noise = random.rand(Np, Ny)
        omega = H + Noise*epsilon
        omega_hat = (fftn_mpi(omega, omega_hat))
        omega = real(ifftn_mpi(omega_hat, omega))
    if ICchoice == 'omega4':
        H = exp(-((2*X - pi + pi / 5) ** 2 + (4*Y - pi + pi / 5) ** 2) / 0.3) - exp(
            -((2*X - pi - pi / 5) ** 2 + (3*Y - pi + pi / 5) ** 2) / 0.2) + exp(
            -((X + pi - pi / 5) ** 2 + (2*Y - pi - pi / 5) ** 2) / 0.4)+exp(-((2*X - pi + pi / 5)**2 + (Y - pi + pi / 5) ** 2) / 0.3) - exp(
            -((X - pi - pi / 5)**2 + (Y - pi + pi / 5) ** 2) /0.2) + exp(-((X - pi - pi / 5)**2 + (3*Y - pi - pi / 5)**2)/0.4)+\
            exp(-((X - pi + pi / 5)**2 + (Y - pi + pi / 5) ** 2) / 0.3) + exp(
            -((X - pi - pi / 5)**2 + (3*Y - pi + pi / 5) ** 2) /0.3) + exp(-((X + pi - pi / 5)**2 + (Y + pi - pi / 5)**2)/0.4)-\
            exp(-((X - pi + pi / 5) ** 2 + (Y - pi + pi / 5) ** 2) / 0.3) + exp(
            -((2*X - pi - pi / 5) ** 2 + (3*Y - pi + pi / 5) ** 2) / 0.2) + exp(
            -((X - pi - pi / 5) ** 2 + (Y - pi - pi / 5) ** 2) / 0.4)
        epsilon = 0.7;
        Noise = random.rand(Np, Ny)
        omega = H + Noise*epsilon
        omega_hat = (fftn_mpi(omega, omega_hat))
        omega = real(ifftn_mpi(omega_hat, omega))
    if ICchoice == 'omega2':

        H = exp(-((X - pi + pi / 5)**2 + (Y - pi + pi / 5) ** 2) / 0.3) - exp(
            -((X - pi - pi / 5)**2 + (Y - pi + pi / 5) ** 2) /0.2) + exp(-((X - pi - pi / 5)**2 + (Y - pi - pi / 5)**2)/0.4);
        #epsilon = 0.1;
        #Noise = random.rand(Np, Ny)
        omega = H
        omega_hat = (fftn_mpi(omega,omega_hat))
        omega = real(ifftn_mpi(omega_hat,omega))
    return omega_hat



# initialize x,y kx, ky coordinate
def IC_coor(Nx, Ny, Np, dx, dy, rank, num_processes):
    x = zeros((Np, Ny), dtype=float);
    y = zeros((Np, Ny), dtype=float);
    kx = zeros((Nx, Np), dtype=float);
    ky = zeros((Nx, Np), dtype=float);
    for j in range(Ny):
        x[0:Np, j] = range(Np);
        if num_processes == 1:
            x[0:Nx, j] = range(int(-Nx / 2), int(Nx / 2));
    # offset for mpi
    if num_processes != 1:
        x = x - (num_processes / 2 - rank) * Np
    x = x * dx;
    for i in range(Np):
        y[i, 0:Ny] = range(int(-Ny / 2), int(Ny / 2));
    y = y * dy;

    for j in range(Np):
        kx[0:Nx, j] = range(int(-Nx / 2), int(Nx / 2));
    for i in range(Nx):
        ky[i, 0:Np] = range(Np);
        if num_processes == 1:
            ky[i, 0:Ny] = range(int(-Ny / 2), int(Ny / 2));
    # offset for mpi
    if num_processes != 1:
        ky = ky - (num_processes / 2 - rank) * Np

    k2 = kx ** 2 + ky ** 2;
    for i in range(Nx):
        for j in range(Np):
            if (k2[i, j] == 0):
                k2[i, j] = 1e-5;  # so that I do not divide by 0 below when using
            # projection operator
    # k2_exp = exp(-nu * (k2 ** 5) * dt - nu_hypo * dt);
    k2_inv = K2_inv = 1 / where(k2 == 0, 1, k2).astype(float)
    return x, y, kx, ky, k2, k2_inv


# -----------GRID setup-----------
Lx = 2 * pi;
Ly = 2 * pi;
dx = Lx / Nx;
dy = Ly / Ny;
x, y, Kx, Ky, K2, K2_inv = IC_coor(Nx, Ny, Np, dx, dy, rank, num_processes)
# TODO check what needs to be done with the wave numbers to get rid of IC_Coor function

sx = slice(rank*Np,(rank+1)*Np)
Xmesh = mgrid[sx, :N].astype(float) * Lx / N
X = Xmesh[0]
Y = Xmesh[1]


x = Y[0]
y = Y[0]
kx = fftfreq(N, 1. / N)
ky = kx.copy()
'''
K = array(meshgrid(kx, ky[sx], indexing='ij'), dtype=int)
Kx = K[1]
Ky = K[0]
K2 = sum(K * K, 0, dtype=int)
'''

LapHat = K2.copy()
LapHat *= -1
#K2[0][0] = 1
K2 *=-1
K2_inv = 1 / where(K2 == 0, 1, K2).astype(float)
ikx_over_K2 = 1j * Kx * K2_inv
iky_over_K2 = 1j * Ky * K2_inv

kmax_dealias = 2. / 3. * (N / 2 + 1)
dealias = array((abs(Kx) < kmax_dealias) * (abs(Ky) < kmax_dealias), dtype=bool)

# ----Initialize Variables-------(hat denotes variables in Fourier space,)
u_hat = zeros((Nx, Np), dtype=complex);
v_hat = zeros((Nx, Np), dtype=complex);
u = zeros((Np, Ny), dtype=float);
v = zeros((Np, Ny), dtype=float);
omega_hat0 = zeros((Nx, Np), dtype=complex);
omega_hat1 = zeros((Nx, Np), dtype=complex);
omega_hat = zeros((Nx, Np), dtype=complex);
omega_hat_new = zeros((Nx, Np), dtype=complex);
omega = zeros((Np, Ny), dtype=float);
omega_kx = zeros((Np, Ny), dtype=float);
omega_ky = zeros((Np, Ny), dtype=float);
v_grad_omega = zeros((Np, Ny), dtype=float);
psi_hat = zeros((Nx, Np), dtype=complex);
rhs_hat = zeros((Nx, Np), dtype=complex);
rhs = zeros((Np, Nx), dtype=float);
visc_term_complex = zeros((Ny, Np), dtype=complex)
visc_term_real = zeros((Np, Ny), dtype=float)
v_grad_omega_hat = zeros((Ny, Np), dtype=complex)

# generate initial velocity field
omega_hat_t0 = IC_condition(Nx, Np, u, v, u_hat, v_hat, ICchoice, omega, omega_hat, X, Y)
omega = ifftn_mpi(omega_hat_t0,omega)

step = 1
try:
    pbar = tqdm(total=int(Nstep))
except:
    pass

# ----Main Loop-----------
for n in range(Nstep + 1):
    if n == 0:
        # TODO check what needs to be done to use IC from matlab program
        # TODO very low convection? bug? compare with animation plot on github
        omega_hat = omega_hat_t0

    u_hat = -iky_over_K2 * omega_hat
    v_hat = ikx_over_K2 * omega_hat
    u = ifftn_mpi(u_hat, u)
    v = ifftn_mpi(v_hat, v)

    omega_kx = ifftn_mpi(1j*Kx * omega_hat, omega_kx)
    omega_ky = ifftn_mpi(1j*Ky * omega_hat, omega_ky)
    v_grad_omega = (u * omega_kx + v * omega_ky)
    v_grad_omega_hat = fftn_mpi(v_grad_omega, v_grad_omega_hat)*dealias
    visc_term_complex = -nu * K2 * omega_hat

    omega_hat_new = 1 / (1 / dt - 0.5 * nu * LapHat)*(
                (1 / dt + 0.5 * nu * LapHat)* omega_hat - v_grad_omega_hat);
    omega_hat = omega_hat_new.copy()

    t = t + dt;
    step += 1
    try:
        pbar.update(1)
    except:
        pass
