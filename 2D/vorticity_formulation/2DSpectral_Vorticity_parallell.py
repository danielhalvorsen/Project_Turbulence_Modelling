# Solver of 2D Navier Stokes equation on streamfunction-vorticity formulation.
# TODO fix matplotlib such that one can plot two figures, one with velocity and one with
# vorticity simultaneously.
# TODO FIX MAPPING STRUCTURE IN FOLDER. DELETE UNECESSARY FILES!!!
from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2
import random
from numpy.random import seed, uniform
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation
from mpi4py import MPI

global u, v


# Base constants and spatial grid vectors
nu = 1e-6
L = pi
N = int(64)
N2 = int(N ** 2)
N_half=int(N/2+1)
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = int(comm.Get_rank())
Np = int(N / num_processes)
Xgrid = mgrid[rank * Np:(rank + 1) * Np, :N].astype(float) * 2 * L / N

# dx = 2 * L / N
# x = linspace(1 - N / 2, N / 2, N) * dx
# y = linspace(1 - N / 2, N / 2, N) * dx
X = Xgrid[0]
Y = Xgrid[1]

U_hat0 = empty((2, Np, N_half), dtype=complex)
U_hat1 = empty((2, Np, N_half), dtype=complex)
dU = empty((2, Np, N_half), dtype=complex)
Uc_hat = empty((Np, N_half), dtype=complex)
Uc_hatT = empty((Np, N_half), dtype=complex)
U_mpi = empty((num_processes, Np, N_half), dtype=complex)
omega = empty((Np,N))
omega_hat = empty((Np,N_half),dtype=complex)

# Spectral frequencies and grid vectors
kx = fftfreq(N, 1. / N)
ky = kx.copy()
K = array(meshgrid(kx, ky), dtype=int)
Kx = K[0]
Ky = K[1]
K2 = sum(K * K, 0, dtype=int)
K2_inv = 1 / where(K2 == 0, 1, K2).astype(float)
K2_inv[0][0] = 0
# Dx = 1j * Kx * K2_inv
# Dy = 1j * Ky * K2_inv
kmax_dealias = 2. / 3. * (N / 2 + 1)
dealias = array(
    (Kx < kmax_dealias) * (Ky < kmax_dealias),
    dtype=bool)


# Initialize solution vector

####################################################################################################
def init_Omega(omega_grid):
    # Set initial condition on the Omega vector in Fourier space
    seed(1969)
    omega_hat = fft.fft2(omega_grid)
    omega_hat[0, 4] = uniform() + 1j * uniform()
    omega_hat[1, 1] = uniform() + 1j * uniform()
    omega_hat[3, 0] = uniform() + 1j * uniform()
    # print(omega_hat)
    omega_IC = real(fft.ifft2(omega_hat))
    omega_IC = omega_IC / max(omega_IC)
    reshaped_omega_IC = reshape(omega_IC, N2, 1)

    return reshaped_omega_IC


def initialize(choice,omega_hat,omega):
    # Set initial condition on the velocity vectors.
    if choice == 'random':
        u = array([[random.random() for i in range(N)] for j in range(N)])
        v = array([[random.random() for i in range(N)] for j in range(N)])
        u_hat = fft.fft2(u)
        v_hat = fft.fft2(v)
        omega_hat = (v_hat * Kx) - (u_hat * Ky)
        omega = real(fft.ifft2(omega_hat))
        omega = omega / max(omega)
        omega_vector = reshape(omega, N2, 1)
    if choice == 'circle':
        x = arange(0, N)
        y = arange(0, N)
        u = ones((y.size, x.size)) * -1
        v = ones((y.size, x.size)) * -1
        cx = N / 2
        cy = N / 2
        r = int(N / 10)
        # The two lines below could be merged, but I stored the mask
        # for code clarity.
        mask = (x[newaxis, :] - cx) ** 2 + (y[:, newaxis] - cy) ** 2 < r ** 2
        u[mask] = 100
        v[mask] = -100
        u_hat = fft.fft2(u)
        v_hat = fft.fft2(v)
        omega_hat = ((v_hat * Kx) - (u_hat * Ky)) * 1j
        omega = real(fft.ifft2(omega_hat))
        omega = omega / max(omega)
        omega_vector = reshape(omega, N2, 1)
    if choice == 'velocity_strips':
        u = ones([N, N]) * 0
        v = ones([N, N]) * 0
        u[int(N / 2 + N / 9 - N / 20):int(N / 2 + N / 9 + N / 20), :] = 1
        u[int(N / 2 - N / 9 - N / 20):int(N / 2 - N / 9 + N / 20), :] = -1
        # plt.imshow(u)
        # plt.show()

        u_hat = fft.fft2(u)
        v_hat = fft.fft2(v)
        omega_hat = ((v_hat * Kx) - (u_hat * Ky)) * 1j
        omega = real(fft.ifft2(omega_hat))
        #  omega = omega / max(omega)
        omega_vector = reshape(omega, N2, 1)

    if choice == 'omega_1':
        seed(1969)
        omega_grid = zeros([Np, N])
        omega_hat = fftn_mpi(omega_grid, omega_hat)
        omega_hat[0, 4] = uniform() + 1j * uniform()
        omega_hat[1, 1] = uniform() + 1j * uniform()
        omega_hat[3, 0] = uniform() + 1j * uniform()
        omega_hat[2, 3] = uniform() + 1j * uniform()
        # print(omega_hat)
        omega = real(ifftn_mpi(omega_hat, omega))
        #omega = omega / max(omega)
        omega_vector = reshape(omega, N2, 1)
    return omega_vector


def Rhs(t, omega_vector):  # change order of arguments for different ode solver
    global u, v
    omega = reshape(omega_vector, ([N, N])).transpose()
    omega_hat = fftn_mpi(omega, omega_hat)  # *dealias
    # print(omega_hat)
    #    omega_hat = multiply(omega_hat,dealias)
    # print(omega_hat)
    omx = real(ifftn_mpi(1j * Kx * omega_hat * dealias, omx))
    omy = real(ifftn_mpi(1j * Ky * omega_hat * dealias, omy))
    psi_hat = omega_hat * K2_inv
    u = real(ifftn_mpi(-1j * Ky * psi_hat * dealias, u))
    v = real(ifftn_mpi(1j * Kx * psi_hat * dealias, v))
    # print(u)
    # u = real(fft.ifft2(Dy * omega_hat * dealias))
    # v = real(fft.ifft2(-Dx * omega_hat * dealias))
    rhs = real(fft.ifft2(-nu * K2 * omega_hat) - u * omx - v * omy)
    # rhs *=dealias
    Rhs = reshape(rhs, N2, 1)
    return Rhs


def writeToFile(solve):
    print('Writing files... ')
    savetxt('dt_vector.txt', solve.t)
    print('Time list written...')
    savetxt('vorticity.txt', solve.y)
    print('Vorticity list written...')
    u_vel, v_vel = convertVorticityToVelocity(solve.y)
    savetxt('u_vel.txt', u_vel)
    print('u velocity list written...')
    savetxt('v_vel.txt', v_vel)
    print('v-velocity list written...')
    # read with: new_data = loadtxt('test.txt')
    print('Finished writing files.')
    return


def convertVorticityToVelocity(solve):
    vorticityField = solve
    u_vel = [None] * len(time_intervals)  # Allocate memory for array
    v_vel = [None] * len(time_intervals)  # Allocate memory for array
    for t in range(len(time_intervals)):
        omega = reshape(vorticityField[:, t], ([N, N])).transpose()
        omega_hat = fft.fft2(omega)
        psi_hat = omega_hat * K2_inv
        u_vel[t] = real(fft.ifft2(-1j * Ky * psi_hat * dealias))
        v_vel[t] = real(fft.ifft2(1j * Kx * psi_hat * dealias))
    u_vel = reshape(array(u_vel), (len(time_intervals), N2), 1)
    v_vel = reshape(array(v_vel), (len(time_intervals), N2), 1)
    return u_vel, v_vel


def ifftn_mpi(fu, u):
    # Inverse Fourier transform
    Uc_hat[:] = ifft(fu)
    comm.Alltoall([Uc_hat, MPI.DOUBLE_COMPLEX], [U_mpi, MPI.DOUBLE_COMPLEX])
    Uc_hatT[:] = rollaxis(U_mpi, 1).reshape(Uc_hatT.shape)
    u[:] = irfft2(Uc_hatT)
    return u


def fftn_mpi(u, fu):
    # Forward Fourier transform
    Uc_hatT[:] = rfft2(u)
    U_mpi[:] = rollaxis(Uc_hatT.reshape(Np, num_processes, N_half), 1)
    comm.Alltoall([U_mpi, MPI.DOUBLE_COMPLEX], [fu, MPI.DOUBLE_COMPLEX])
    fu[:] = fft(fu)
    return fu


####################################################################################################


omega_vector = initialize('omega_1',omega_hat,omega)


animateOmega = False
animateVelocity = False
if (animateOmega or animateVelocity) == True:
    # Temporal data for animation
    t0 = 0
    t_end = 15
    dt = 0.1

    fig = plt.figure()
    numsteps = ceil(t_end / dt)
    step = 1
    pbar = tqdm(total=int(t_end / dt))

    ims = []
    while step <= numsteps:
        solve = integrate.solve_ivp(Rhs, [0, dt], omega_vector, method='RK45', rtol=1e-10,
                                    atol=1e-10)
        omega_vector = solve.y[:, -1]

        if animateOmega == True:
            if step % 5 == 0:
                omega = reshape(omega_vector, ([N, N])).transpose()
                im = plt.imshow(omega, cmap='jet', vmax=1, vmin=-1,
                                animated=True)
                plt.pause(0.05)
                ims.append([im])
        if animateVelocity == True:
            if step % 3 == 0:
                im = plt.imshow(abs((u ** 2) + (v ** 2)), cmap='jet', animated=True)
                ims.append([im])
                plt.pause(0.05)
        step += 1
        pbar.update(1)

    if animateOmega == True:
        cbar = plt.colorbar(im)
        # ScalarMappable.set_clim(min_val,max_val)
        # cbar.set_ticks(linspace(min_val,max_val,10))
        cbar.set_label('Vorticity magnitude [m/s]')
        plt.xlim(0, N)
        plt.ylim(0, N)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        # plt.axes().set_aspect('equal')
        ani = animation.ArtistAnimation(fig, ims, interval=15, blit=True,
                                        repeat_delay=None)
        ani.save('animation_folder/animationVorticity.gif', writer='imagemagick', fps=30)
        plt.show()
    if animateVelocity == True:
        cbar = plt.colorbar(im)
        cbar.set_label('Velocity magnitude [m/s]')
        plt.xlim(0, N)
        plt.ylim(0, N)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        # plt.axes().set_aspect('equal')
        ani = animation.ArtistAnimation(fig, ims, interval=15, blit=True,
                                        repeat_delay=None)
        ani.save('animation_folder/animationVelocity.gif', writer='imagemagick', fps=30)
        # TODO find out what interval we need to make a gif of a certain length in
        #  seconds.
        plt.show()

    pbar.close()

if (animateVelocity and animateOmega) == False:
    # Temporal data for non-animation
    print('Entering false script')
    # TODO for verification, the maximum dt can be changed in the ODE option argument
    t0 = 0
    t_end = 15
    dt = 0.1
    time_intervals = linspace(t0, t_end, t_end / dt + 1)
    solve = integrate.solve_ivp(Rhs, [0, t_end], omega_vector, method='RK45',
                                t_eval=time_intervals, rtol=1e-10,
                                atol=1e-10)
    plt.imshow(abs((u ** 2) + (v ** 2)), cmap='jet')
    plt.show()
    writeToFile(solve)  # Written to work for integrate.solve_ivp().
print(' ------- Script finished -------')
