# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:14:22 2016

@author: Xin
"""
# solve 2-D incompressible NS equations using spectral method
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os.path
from numpy import *
from numpy import max as npmax
from numpy.fft import fftfreq, fft, ifft, fft2, ifft2, fftshift, ifftshift
from mpi4py import MPI
from tqdm import tqdm
import matplotlib.animation as animation

parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(parent)

# parameters
new = 1;
Nstep = 2500000;  # no. of steps
N = Nx = Ny = 64;  # grid size
t = 0;
nu = 1e-4;  # viscosity
#nu_hypo = 2e-3;  # hypo-viscosity
dt = 1e-3;  # time-step
#dt_h = dt / 2;  # half-time step
ic_type = 2  # 1 for Taylor-Green init_cond; 2 for random init_cond
k_ic = 1;  # initial wavenumber for Taylor green forcing
diag_out_step = 0.02*Nstep;  # frequency of outputting diagnostics

fig = plt.figure()
ims = []

# ------------MPI setup---------
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
# slab decomposition, split arrays in x direction in physical space, in ky direction in
# Fourier space
Np = int(N / num_processes)

# ---------declare functions that will be used----------

# ---------2D FFT and IFFT-----------
Uc_hat = empty((N, Np), dtype=complex)
Uc_hatT = empty((Np, N), dtype=complex)
U_mpi = empty((num_processes, Np, Np), dtype=complex)


# inverse FFT
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
    k2_inv = 1 / where(k2 == 0, 1, k2).astype(float)
    #k2_exp = exp(-nu * (k2 ** 5) * dt - nu_hypo * dt);
    return x, y, kx, ky, k2,k2_inv


# ---------Dealiasing function----
def delias(u_hat, v_hat, Nx, Np, k2):
    # use 1/3 rule to remove values of wavenumber >= Nx/3
    for i in range(Nx):
        for j in range(Np):
            if (sqrt(k2[i, j]) >= Nx / 3.):
                u_hat[i, j] = 0;
                v_hat[i, j] = 0;
    # Projection operator on velocity fields to make them solenoidal-----
    tmp = (kx * u_hat + ky * v_hat) / k2;
    u_hat = u_hat - kx * tmp;
    v_hat = v_hat - ky * tmp;
    return u_hat, v_hat


# ----Initialize Velocity in Fourier space-----------
def IC_condition(ic_type, k_ic, kx, ky, Nx, Np):
    # taylor green vorticity field
    u_hat = zeros((Nx, Np), dtype=complex);
    v_hat = zeros((Nx, Np), dtype=complex);
    if (new == 1 and ic_type == 1):
        for iss in [-1, 1]:
            for jss in [-1, 1]:
                for i in range(Nx):
                    for j in range(Np):
                        if (int(kx[i, j]) == iss * k_ic and int(ky[i, j]) == jss * k_ic):
                            u_hat[i, j] = -1j * iss;
                            v_hat[i, j] = -1j * (-jss);
        # Set total energy to 1
        u_hat = 0.5 * u_hat;
        v_hat = 0.5 * v_hat;
    # generate random velocity field
    elif (new == 1 and ic_type == 2):
        u = random.rand(Np, Ny)
        v = random.rand(Np, Ny)
        u_hat = fftn_mpi(u, u_hat)
        v_hat = fftn_mpi(v, v_hat)
    return u_hat, v_hat


# ------output function----
# this function output vorticty contour
def output(omega, x, y, Nx, Ny, rank, time, plotstring):
    # collect values to root
    omega_all = comm.gather(omega, root=0)
    u_all = comm.gather(u, root=0)
    v_all = comm.gather(v, root=0)
    x_all = comm.gather(x, root=0)
    y_all = comm.gather(y, root=0)
    if rank == 0:
        if plotstring == 'Vorticity':
            # reshape the ''list''
            omega_all = asarray(omega_all).reshape(Nx, Ny)
            x_all = asarray(x_all).reshape(Nx, Ny)
            y_all = asarray(y_all).reshape(Nx, Ny)
            plt.contourf(x_all, y_all, omega_all, cmap='jet')
            delimiter = ''
            title = delimiter.join(['vorticity contour, time=', str(time)])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(title)
            plt.show()
            filename = delimiter.join(['vorticity_t=', str(time), '.png'])
            plt.savefig(filename, format='png')
        if plotstring == 'Velocity':
            # reshape the ''list''
            u_all = asarray(u_all).reshape(Nx, Ny)
            v_all = asarray(v_all).reshape(Nx, Ny)
            x_all = asarray(x_all).reshape(Nx, Ny)
            y_all = asarray(y_all).reshape(Nx, Ny)
            #plt.contourf(x_all, y_all, (u_all ** 2 + v_all ** 2), cmap='jet')
            plt.imshow((u_all ** 2 + v_all ** 2), cmap='jet')
            delimiter = ''
            title = delimiter.join(['Velocity contour, time=', str(time)])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(title)
            plt.show()
            filename = delimiter.join(['velocity_t=', str(time), '.png'])
            plt.savefig(filename, format='png')
        if plotstring == 'VelocityAnimation':
            # reshape the ''list''
            u_all = asarray(u_all).reshape(Nx, Ny)
            v_all = asarray(v_all).reshape(Nx, Ny)
            im = plt.imshow(abs((u_all ** 2) + (v_all ** 2)), cmap='jet', animated=True)
            ims.append([im])
        if plotstring == 'VorticityAnimation':
            omega_all = asarray(omega_all).reshape(Nx, Ny)
            im = plt.imshow(omega_all, cmap='jet', animated=True)
            ims.append([im])
# --------------finish declaration of functions-------


# -----------GRID setup-----------
Lx = 2 * pi;
Ly = 2 * pi;
dx = Lx / Nx;
dy = Ly / Ny;

# obtain x, y, kx, ky
x, y, kx, ky, k2,k2_inv = IC_coor(Nx, Ny, Np, dx, dy, rank, num_processes)

kmax_dealias = 2. / 3. * (N / 2 + 1)
dealias = array(
    (kx < kmax_dealias) * (ky < kmax_dealias),
    dtype=bool)
# ----Initialize Variables-------(hat denotes variables in Fourier space,)
# velocity
u_hat = zeros((Nx, Np), dtype=complex);
v_hat = zeros((Nx, Np), dtype=complex);
# Vorticity
omega_hat = zeros((Nx, Np), dtype=complex);
# Nonlinear term
NLxhat = zeros((Nx, Np), dtype=complex);
NLyhat = zeros((Nx, Np), dtype=complex);
rhs = zeros((Nx,Np),dtype=complex);
psi_hat = zeros((Nx,Np),dtype=complex);
visc_term_complex = zeros((Ny,Np),dtype=complex)
# variables in physical space
u = zeros((Np, Ny), dtype=float);
v = zeros((Np, Ny), dtype=float);
omega = zeros((Np, Ny), dtype=float);
omx = zeros((Np,Ny),dtype=float);
omy = zeros((Np,Ny),dtype=float);
visc_term_real = zeros((Np,Ny),dtype=float)

# generate initial velocity field
u_hat, v_hat = IC_condition(ic_type, k_ic, kx, ky, Nx, Np)

# ------Dealiasing------------------------------------------------
#u_hat, v_hat = delias(u_hat, v_hat, Nx, Np, k2)
#
# ------Storing variables for later use in time integration--------
omega_hat_t0 = 1j * (kx * v_hat - ky * u_hat);

#

step = 1
pbar = tqdm(total=int(Nstep))

# ----Main Loop-----------
for n in range(Nstep + 1):
    # ------Dealiasing
 #   u_hat, v_hat = delias(u_hat, v_hat, Nx, Np, k2)
    if n==0:
        omega_hat = omega_hat_t0
        ''' random.seed(1969)
        omega_hat[0, 4] = random.uniform() + 1j * random.uniform()
        omega_hat[1, 1] = random.uniform() + 1j * random.uniform()
        omega_hat[3, 0] = random.uniform() + 1j * random.uniform()
        omega_hat[2, 3] = random.uniform() + 1j * random.uniform()
        omega = ifftn_mpi(omega_hat,omega)
        omega = omega/npmax(omega)
        omega_hat = fftn_mpi(omega,omega_hat)
        print('using first omega') 
        '''
    else:
        omega_hat = fftn_mpi(omega,omega_hat)
    #omega_hat = fftn_mpi(omega, omega_hat)
    psi_hat = omega_hat * k2_inv
    u = ifftn_mpi(-1j*ky*psi_hat,u)
    v = ifftn_mpi(1j*kx*psi_hat,v)


    # Calculate non-linear term in x-space
    omx = ifftn_mpi(1j*kx*omega_hat*dealias,omx)
    omy = ifftn_mpi(1j*ky*omega_hat*dealias,omy)
    visc_term_complex = -nu*k2*omega_hat
    visc_term_real    = ifftn_mpi(visc_term_complex,visc_term_real)

    rhs = visc_term_real-u*omx-v*omy

    # Integrate in time
    # ---Euler for 1/2-step-----------
    omega = omega + dt * rhs

    # output vorticity contour
    if (n % diag_out_step == 0):
        output(omega, x, y, Nx, Ny, rank, t, 'VelocityAnimation')
    t = t + dt;
    step += 1
    pbar.update(1)
ani = animation.ArtistAnimation(fig, ims, interval=15, blit=True,
                                        repeat_delay=None)
ani.save('animationVelocity.gif', writer='imagemagick', fps=30)
