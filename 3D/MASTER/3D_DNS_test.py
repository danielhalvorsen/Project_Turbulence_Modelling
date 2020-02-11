from numpy import *
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2,fftn,fftshift,rfft
from mpi4py import MPI
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation
import time


# U is set to dtype float32
# Reynoldsnumber determined by nu Re = 1600, nu = 1/1600
nu = 0.0000000625
# nu = 0.00000625
T = 40
dt = 0.01
N = int(2 ** 6)
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
dir = '/home/danieloh/PycharmProjects/Project_Turbulence_Modelling/animation_folder/'

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


# initial condition and transformation to Fourier space
U[0] = sin(X[0]) * cos(X[1]) * cos(X[2])
U[1] = -cos(X[0]) * sin(X[1]) * cos(X[2])
U[2] = 0
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


    if tstep==30:
        u_plot = comm.gather(U, root=0)
        if rank==0:
            U_test = concatenate(u_plot, axis=1)
            if plotting == 'animation':
                im = plt.imshow(U_test[0][:,:,int(N/2)],cmap='jet', animated=True)
                ims.append([im])
            if plotting == 'plot':
                plt.imshow(U_test[0][:,:,int(N/2)],cmap='jet')
                plt.pause(0.05)
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