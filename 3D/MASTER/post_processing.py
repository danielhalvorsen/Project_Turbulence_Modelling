import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.fft import fftfreq, fft, ifft, irfft2, fftn,fftshift,rfft,irfft
import time
from numpy import sqrt, zeros, conj, pi, arange, ones, convolve
from matplotlib.ticker import ScalarFormatter
from numba import jit

def movingaverage(interval, window_size):
    window = ones(int(window_size)) / float(window_size)
    return convolve(interval, window, 'same')

@jit(nopython=True)
def kloop(nx,ny,nz,tke_spectrum,tkeh):
    for kx in range(-nx//2, nx//2-1):
        for ky in range(-ny//2, ny//2-1):
            for kz in range(-nz//2, nz//2-1):
                rk = sqrt(kx**2 + ky**2 + kz**2)
                k = int(np.round(rk))
                tke_spectrum[k] += tkeh[kx, ky, kz]
    return tke_spectrum

def compute_tke_spectrum(u, v, w, N, smooth):
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
    lx,ly,lz = N,N,N
    nx = len(u[:, 0, 0])
    ny = len(v[0, :, 0])
    nz = len(w[0, 0, :])

    nt = nx * ny * nz
    n = nx  # int(np.round(np.power(nt,1.0/3.0)))

    uh = fftn(u) / nt
    vh = fftn(v) / nt
    wh = fftn(w) / nt

    tkeh = 0.5 * (uh * conj(uh) + vh * conj(vh) + wh * conj(wh)).real

    k0x = 2.0 * pi / lx
    k0y = 2.0 * pi / ly
    k0z = 2.0 * pi / lz

    knorm = (k0x + k0y + k0z) / 3.0
    print('knorm = ', knorm)

    kxmax = nx / 2
    kymax = ny / 2
    kzmax = nz / 2

    # dk = (knorm - kmax)/n
    # wn = knorm + 0.5 * dk + arange(0, nmodes) * dk

    wave_numbers = knorm * arange(0, n)

    tke_spectrum = zeros(len(wave_numbers))

    tke_spectrum = kloop(nx,ny,nz,tke_spectrum,tkeh)

    # for kx in range(nx):
    #     rkx = kx
    #     if kx > kxmax:
    #         rkx = rkx - nx
    #     for ky in range(ny):
    #         rky = ky
    #         if ky > kymax:
    #             rky = rky - ny
    #         for kz in range(nz):
    #             rkz = kz
    #             if kz > kzmax:
    #                 rkz = rkz - nz
    #             rk = sqrt(rkx * rkx + rky * rky + rkz * rkz)
    #             k = int(np.round(rk))
    #             tke_spectrum[k] = tke_spectrum[k] + tkeh[kx, ky, kz]

    tke_spectrum = tke_spectrum / knorm

    #  tke_spectrum = tke_spectrum[1:]
    #  wave_numbers = wave_numbers[1:]
    if smooth:
        tkespecsmooth = movingaverage(tke_spectrum, 5)  # smooth the spectrum
        tkespecsmooth[0:4] = tke_spectrum[0:4]  # get the first 4 values from the original data
        tke_spectrum = tkespecsmooth

    knyquist = knorm * min(nx, ny, nz) / 2

    return knyquist, wave_numbers, tke_spectrum

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

    sign_sizex = np.shape(EK_U)[0]
    sign_sizey = np.shape(EK_U)[1]
    sign_sizez = np.shape(EK_U)[2]

    box_sidex = sign_sizex
    box_sidey = sign_sizey
    box_sidez = sign_sizez

    box_radius = int(np.ceil((np.sqrt((box_sidex) ** 2 + (box_sidey) ** 2 + (box_sidez) ** 2)) / 2.) + 1)

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

    EK_U_avsphr = np.zeros(box_radius, ) + eps  ## size of the radius
    EK_V_avsphr = np.zeros(box_radius, ) + eps  ## size of the radius
    EK_W_avsphr = np.zeros(box_radius, ) + eps  ## size of the radius

    for i in range(box_sidex):
        for j in range(box_sidey):
            for k in range(box_sidez):
                wn = int(round(np.sqrt((i - centerx) ** 2 + (j - centery) ** 2 + (k - centerz) ** 2)))
                EK_U_avsphr[wn] = EK_U_avsphr[wn] + EK_U[i, j, k]
                EK_V_avsphr[wn] = EK_V_avsphr[wn] + EK_V[i, j, k]
                EK_W_avsphr[wn] = EK_W_avsphr[wn] + EK_W[i, j, k]
        print('iterating'+str(i),flush=True)

    EK_avsphr = 0.5 * (EK_U_avsphr + EK_V_avsphr + EK_W_avsphr)

    fig2 = plt.figure()
    #plt.title("Kinetic Energy Spectrum")
    plt.xlabel(r"k")
    plt.ylabel(r"E(k)")

    realsize = len(rfft(U[:, 0, 0]))
    plt.loglog(np.arange(0, realsize), ((EK_avsphr[0:realsize])), 'k')
    plt.loglog(np.arange(realsize, len(EK_avsphr), 1), ((EK_avsphr[realsize:])), 'k--')
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

    dataout = np.zeros((box_radius, 2))
    dataout[:, 0] = np.arange(0, len(dataout))
    dataout[:, 1] = EK_avsphr[0:len(dataout)]

    #savetxt(Figs_Path + Fig_file_name + '.dat', dataout)
    #fig.savefig(Figs_Path + Fig_file_name + '.pdf')
    return fig2

#TODO make spectrum plots animated and add to github




fig, ax = plt.subplots()

ims = []
step=0
xticks = np.logspace(0,3,5)
yticks = np.logspace(1,-13,7)
N=512
amount = 32
name = 'vel_files/velocity_'+str(step)+'.npy'
plot = 'spectrum2'

#TODO load in one and one file from /vel_files, read [0][:,:,-1] and add to animation. Also make spectrum plots and viscous diffusion plots
for i in range(amount):
    name = 'vel_files/velocity_' + str(step) + '.npy'
    vec = np.load(name)
    print('Loaded nr: '+str(step),flush=True)
    if plot == 'plotVelocity':
        im = plt.imshow(vec[0][:,:,-1],cmap='jet', animated=True)
        ims.append([im])
    if plot == 'spectrum1':
        fig = spectrum(N,vec[0],vec[1],vec[2])
        plt.savefig('spectrum_plots/spectrum_'+str(step))
        #im = plt.show()
        #ims.append([im])
    if plot == 'spectrum2':
        nyquist,k,tke = compute_tke_spectrum(vec[0],vec[1],vec[2],N,True)
        plt.loglog(k,tke,'k-')
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlabel('Wave number, $k$')
        plt.ylabel('Turbulent kinetic energy, $E(k)$')
        plt.savefig('spectrum_plots/spectrum_'+str(step))
        plt.clf()

    step += 140
    print('Finished appending nr: '+str(step),flush=True)

'''
if plot =='spectrum2':
    ani = animation.ArtistAnimation(fig, ims, interval=2, blit=False,repeat_delay=None)
    ani.save('spectrum.gif', writer='imagemagick')
'''