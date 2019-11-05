# solve 2-D incompressible NS equations using spectral method

import matplotlib.pyplot as plt
from numpy import *
from numpy import max as npmax
from numpy.fft import fftfreq, fft, ifft, fft2, ifft2, fftshift, ifftshift
from mpi4py import MPI
from tqdm import tqdm
import matplotlib.animation as animation

#parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
#sys.path.append(parent)

# parameters
tend = 10
dt = 1e-3
Nstep = int(ceil(tend / dt))
N = Nx = Ny = 64;  # grid size
t = 0;
nu = 1e-2;  # viscosity
aniNr = 0.02*Nstep
save_dt = 1e-4
save_every = Nstep * save_dt
save_interval = int(ceil(Nstep / save_every))
t_list = linspace(0,tend,1/save_dt+1)

# ------------MPI setup---------
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
Np = int(N / num_processes)
# slab decomposition, split arrays in x direction in physical space, in ky direction in
# Fourier space
Uc_hat = empty((N, Np), dtype=complex)
Uc_hatT = empty((Np, N), dtype=complex)
U_mpi = empty((num_processes, Np, Np), dtype=complex)

a = [1./6.,1./3.,1./3.,1./6.]
b = [0.5,0.5,1.]
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


# ----Initialize Velocity in Fourier space-----------
def IC_condition(Nx, Np):
    # taylor green vorticity field
    u_hat = zeros((Nx, Np), dtype=complex);
    v_hat = zeros((Nx, Np), dtype=complex);
    # generate random velocity field
    u = random.rand(Np, Ny)
    v = random.rand(Np, Ny)
    u_hat = fftn_mpi(u, u_hat)
    v_hat = fftn_mpi(v, v_hat)
    return u_hat, v_hat


# ------output function----
# this function output vorticty contour
def output(save_counter, omega, x, y, Nx, Ny, rank, time, plotstring):
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
            # plt.contourf(x_all, y_all, (u_all ** 2 + v_all ** 2), cmap='jet')
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
        if plotstring == 'store':
            omega_all = asarray(omega_all).reshape(Nx, Ny)
            u_all = asarray(u_all).reshape(Nx, Ny)
            v_all = asarray(v_all).reshape(Nx, Ny)

            u_storage[save_counter] = u_all
            v_storage[save_counter] = v_all
            omega_storage[save_counter] = omega_all

# -----------GRID setup-----------
Lx = 2 * pi;
Ly = 2 * pi;
dx = Lx / Nx;
dy = Ly / Ny;

Xmesh = mgrid[rank * Np:(rank + 1) * Np, :N].astype(float) * Lx / N
X = Xmesh[0]
Y = Xmesh[1]
x = Y[0]
y = Y[0]
kx = fftfreq(N, 1. / N)
ky = kx.copy()
K = array(meshgrid(kx, ky[rank * Np:(rank + 1) * Np], indexing='ij'), dtype=int)
Kx = K[0]
Ky = K[1]
K2 = sum(K * K, 0, dtype=int)
K2[0][0] = 1
K2_inv = 1 / where(K2 == 0, 1, K2).astype(float)


kmax_dealias = 2. / 3. * (N / 2 + 1)
dealias = array((Kx < kmax_dealias) * (Ky < kmax_dealias),dtype=bool)

# ----Initialize Variables-------(hat denotes variables in Fourier space,)
u_hat = zeros((Nx, Np), dtype=complex);
v_hat = zeros((Nx, Np), dtype=complex);
u = zeros((Np, Ny), dtype=float);
v = zeros((Np, Ny), dtype=float);
omega_hat0 = zeros((Nx, Np), dtype=complex);
omega_hat1 = zeros((Nx, Np), dtype=complex);
omega_hat = zeros((Nx, Np), dtype=complex);
omega = zeros((Np, Ny), dtype=float);
omega_kx = zeros((Np, Ny), dtype=float);
omega_ky = zeros((Np, Ny), dtype=float);
v_grad_omega = zeros((Np, Ny), dtype=float);
psi_hat = zeros((Nx, Np), dtype=complex);
visc_term_complex = zeros((Ny, Np), dtype=complex)
v_grad_omega_hat = zeros((Ny, Np), dtype=complex)
u_storage = empty((save_interval + 1, Nx, Nx), dtype=float)
v_storage = empty((save_interval + 1, Nx, Nx), dtype=float)
omega_storage = empty((save_interval + 1, Nx, Nx), dtype=float)
# generate initial velocity field
u_hat, v_hat = IC_condition(Nx, Np)*dealias


omega_hat_t0 = 1j * (Kx * v_hat - Ky * u_hat);
step = 1
pbar = tqdm(total=int(Nstep))
save_counter = 0
plotstring=('VelocityAnimation')
fig = plt.figure()
ims = []

# ----Main Loop-----------
for n in range(Nstep + 1):
    if n == 0:
        omega_hat = omega_hat_t0

    psi_hat = -omega_hat * K2_inv
    u_hat = 1j * Ky * psi_hat
    v_hat = -1j * Kx * psi_hat
    #u = real(ifftn_mpi(u_hat,u))
    #v = real(ifftn_mpi(v_hat,v))
    #omega_kx =  ifftn_mpi(Kx*omega_hat,omega_kx)
    #omega_ky =  ifftn_mpi(Ky*omega_hat,omega_ky)
    v_grad_omega = u*omega_kx+v*omega_ky
    v_grad_omega_hat = fftn_mpi(v_grad_omega,v_grad_omega_hat)*dealias
#    v_grad_omega_hat *= dealias
    visc_term_complex = -nu * K2 * omega_hat


    rhs_hat = visc_term_complex - u_hat * 1j * Kx * omega_hat * dealias - v_hat * 1j * \
              Ky * omega_hat * dealias
    #rhs_hat = visc_term_complex -v_grad_omega_hat


    omega_hat1 = omega_hat0 = omega_hat
    for rk in range(4):
        if rk<3:omega_hat=omega_hat0+b[rk]*dt*rhs_hat
        omega_hat1+=a[rk]*dt*rhs_hat
    omega_hat = omega_hat1



    #omega_hat = omega_hat + dt * rhs_hat
    '''
    if n%200==0:
        plt.imshow(sqrt(u**2+v**2),cmap='jet')
        plt.pause(0.05)
    '''
    if (n % aniNr == 0):
        u = ifftn_mpi(u_hat, u)
        v = ifftn_mpi(v_hat, v)
        omega = ifftn_mpi(omega_hat, omega)
        output(save_counter, omega, x, y, Nx, Ny, rank, t, plotstring)
        save_counter += 1
    t = t + dt;
    step += 1
    pbar.update(1)
if plotstring in ['VelocityAnimation','VorticityAnimation']:
    ani = animation.ArtistAnimation(fig, ims, interval=15, blit=True,repeat_delay=None)
    ani.save('animationVelocity.gif', writer='imagemagick', fps=30)
if plotstring=='store':
    save('datafiles/u_vel', u_storage)
    save('datafiles/v_vel', v_storage)
    save('datafiles/omega', omega_storage)
    save('datafiles/tlist',t_list)
