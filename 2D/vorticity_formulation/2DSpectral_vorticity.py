# Solver of 2D Navier Stokes equation on streamfunction-vorticity formulation.
# TODO fix matplotlib such that one can plot two figures, one with velocity and one with
# vorticity simultaneously.
# TODO FIX MAPPING STRUCTURE IN FOLDER. DELETE UNECESSARY FILES!!!
import numpy as np
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2
import random
from numpy.random import seed, uniform
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation
from numba import jit

global u, v


####################################################################################################
def init_Omega(omega_grid):
    # Set initial condition on the Omega vector in Fourier space
    seed(1969)
    omega_hat = np.fft.fft2(omega_grid)
    omega_hat[0, 4] = uniform() + 1j * uniform()
    omega_hat[1, 1] = uniform() + 1j * uniform()
    omega_hat[3, 0] = uniform() + 1j * uniform()
    # print(omega_hat)
    omega_IC = np.real(np.fft.ifft2(omega_hat))
    omega_IC = omega_IC / np.max(omega_IC)
    reshaped_omega_IC = np.reshape(omega_IC, N2, 1)

    return reshaped_omega_IC


def initialize(choice):
    # Set initial condition on the velocity vectors.
    if choice == 'random':
        u = np.array([[random.random() for i in range(N)] for j in range(N)])
        v = np.array([[random.random() for i in range(N)] for j in range(N)])
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        omega_hat = (v_hat * Kx) - (u_hat * Ky)
        omega = np.real(np.fft.ifft2(omega_hat))
        omega = omega / np.max(omega)
        omega_vector = np.reshape(omega, N2, 1)
    if choice == 'circle':
        x = np.arange(0, N)
        y = np.arange(0, N)
        u = np.ones((y.size, x.size)) * -1
        v = np.ones((y.size, x.size)) * -1
        cx = N / 2
        cy = N / 2
        r = int(N / 10)
        # The two lines below could be merged, but I stored the mask
        # for code clarity.
        mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
        u[mask] = 100
        v[mask] = -100
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        omega_hat = ((v_hat * Kx) - (u_hat * Ky)) * 1j
        omega = np.real(np.fft.ifft2(omega_hat))
        omega = omega / np.max(omega)
        omega_vector = np.reshape(omega, N2, 1)
    if choice == 'velocity_strips':
        u = np.ones([N, N]) * 0
        v = np.ones([N, N]) * 0
        u[int(N / 2 + N / 9 - N / 20):int(N / 2 + N / 9 + N / 20), :] = 1
        u[int(N / 2 - N / 9 - N / 20):int(N / 2 - N / 9 + N / 20), :] = -1
        #plt.imshow(u)
        #plt.show()

        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        omega_hat = ((v_hat * Kx) - (u_hat * Ky)) * 1j
        omega = np.real(np.fft.ifft2(omega_hat))
      #  omega = omega / np.max(omega)
        omega_vector = np.reshape(omega, N2, 1)

    if choice == 'omega_1':
        seed(1969)
        omega_grid = np.zeros([N, N])
        omega_hat = np.fft.fft2(omega_grid)
        omega_hat[0, 4] = uniform() + 1j * uniform()
        omega_hat[1, 1] = uniform() + 1j * uniform()
        omega_hat[3, 0] = uniform() + 1j * uniform()
        omega_hat[2, 3] = uniform() + 1j * uniform()
        # print(omega_hat)
        omega = np.real(np.fft.ifft2(omega_hat))
        omega = omega / np.max(omega)
        omega_vector = np.reshape(omega, N2, 1)
    return omega_vector

#@jit
def Rhs(t, omega_vector):  # change order of arguments for different ode solver
    global u, v
    omega = np.reshape(omega_vector, ([N, N])).transpose()
    omega_hat = np.fft.fft2(omega)  # *dealias
    # print(omega_hat)
    #    omega_hat = np.multiply(omega_hat,dealias)
    # print(omega_hat)
    omx = np.real(np.fft.ifft2(1j * Kx * omega_hat * dealias))
    omy = np.real(np.fft.ifft2(1j * Ky * omega_hat * dealias))
    psi_hat = omega_hat * K2_inv
    u = np.real(np.fft.ifft2(-1j * Ky * psi_hat * dealias))
    v = np.real(np.fft.ifft2(1j * Kx * psi_hat * dealias))
    # print(u)
    # u = np.real(np.fft.ifft2(Dy * omega_hat * dealias))
    # v = np.real(np.fft.ifft2(-Dx * omega_hat * dealias))
    rhs = np.real(np.fft.ifft2(-nu * K2 * omega_hat) - u * omx - v * omy)
    # rhs *=dealias
    Rhs = np.reshape(rhs, N2, 1)
    return Rhs


def writeToFile(solve):
    print('Writing files... ')
    np.save('dt_vector', solve.t)
    print('Time list written...')
    np.save('vorticity', solve.y)
    print('Vorticity list written...')
    u_vel, v_vel = convertVorticityToVelocity(solve.y)
    np.save('u_vel', u_vel)
    print('u velocity list written...')
    np.save('v_vel', v_vel)
    print('v-velocity list written...')
    # read with: new_data = np.loadtxt('test.txt')
    print('Finished writing files.')
    return


def convertVorticityToVelocity(solve):
    vorticityField = solve
    u_vel = [None] * len(time_intervals)  # Allocate memory for array
    v_vel = [None] * len(time_intervals)  # Allocate memory for array
    for t in range(len(time_intervals)):
        omega = np.reshape(vorticityField[:, t], ([N, N])).transpose()
        omega_hat = np.fft.fft2(omega)
        psi_hat = omega_hat * K2_inv
        u_vel[t] = np.real(np.fft.ifft2(-1j * Ky * psi_hat * dealias))
        v_vel[t] = np.real(np.fft.ifft2(1j * Kx * psi_hat * dealias))
    u_vel = np.reshape(np.array(u_vel), (len(time_intervals), N2), 1)
    v_vel = np.reshape(np.array(v_vel), (len(time_intervals), N2), 1)
    return u_vel, v_vel


####################################################################################################

# Base constants and spatial grid vectors
nu = 1e-6
L = np.pi
N = int(256)
N2 = int(N ** 2)
dx = 2 * L / N
x = np.linspace(1 - N / 2, N / 2, N) * dx
y = np.linspace(1 - N / 2, N / 2, N) * dx
[X, Y] = np.meshgrid(x, y)

# Spectral frequencies and grid vectors
kx = fftfreq(N, 1. / N)
ky = kx.copy()
K = np.array(np.meshgrid(kx, ky), dtype=int)
Kx = K[0]
Ky = K[1]
K2 = np.sum(K * K, 0, dtype=int)
K2_inv = 1 / np.where(K2 == 0, 1, K2).astype(float)
K2_inv[0][0] = 0
# Dx = 1j * Kx * K2_inv
# Dy = 1j * Ky * K2_inv
kmax_dealias = 2. / 3. * (N / 2 + 1)
dealias = np.array(
    (Kx < kmax_dealias) * (Ky < kmax_dealias),
    dtype=bool)

# Initialize solution vector
omega_vector = initialize('omega_1')

animateOmega = False
animateVelocity = True
if (animateOmega or animateVelocity) == True:
    # Temporal data for animation
    t0 = 0
    t_end = 15
    dt = 0.1

    fig = plt.figure()
    numsteps = np.ceil(t_end / dt)
    step = 1
    pbar = tqdm(total=int(t_end / dt))

    ims = []
    while step <= numsteps:
        solve = integrate.solve_ivp(Rhs, [0, dt], omega_vector, method='RK45', rtol=1e-10,
                                    atol=1e-10)
        omega_vector = solve.y[:, -1]

        if animateOmega == True:
            if step % 5 == 0:
                omega = np.reshape(omega_vector, ([N, N])).transpose()
                im = plt.imshow(omega, cmap='jet', vmax=1, vmin=-1,
                                animated=True)
                plt.pause(0.05)
                ims.append([im])
        if animateVelocity == True:
            if step % 3 == 0:
                im = plt.imshow(np.abs((u ** 2) + (v ** 2)), cmap='jet', animated=True)
                ims.append([im])
                plt.pause(0.05)
        step += 1
        pbar.update(1)

    if animateOmega == True:
        cbar = plt.colorbar(im)
        # ScalarMappable.set_clim(min_val,max_val)
        # cbar.set_ticks(np.linspace(min_val,max_val,10))
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
    time_intervals = np.linspace(t0,t_end,t_end/dt+1)
    solve = integrate.solve_ivp(Rhs, [0, t_end], omega_vector, method='RK45',
                                t_eval=time_intervals, rtol=1e-10,
                                atol=1e-10)
    plt.imshow(np.abs((u ** 2) + (v ** 2)), cmap='jet')
    plt.show()
    writeToFile(solve)  # Written to work for integrate.solve_ivp().
print(' ------- Script finished -------')
