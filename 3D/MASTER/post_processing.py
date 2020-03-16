import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.fft import fftfreq, fft, ifft, irfft2, fftn,fftshift,rfft,irfft
import time
from numpy import sqrt, zeros, conj, pi, arange, ones, convolve
from matplotlib.ticker import ScalarFormatter
from numba import jit
from mpistuff.mpibase import work_arrays
work_array = work_arrays()





def movingaverage(interval, window_size):
    window = ones(int(window_size)) / float(window_size)
    return convolve(interval, window, 'same')

@jit(nopython=True,parallel=True)
def kloop(nx,ny,nz,tke_spectrum,tkeh):
    for kx in range(-nx//2, nx//2-1):
        for ky in range(-ny//2, ny//2-1):
            for kz in range(-nz//2, nz//2-1):
                rk = sqrt(kx**2 + ky**2 + kz**2)
                k = int(np.round(rk))
                tke_spectrum[k] += tkeh[kx, ky, kz]
    return tke_spectrum

@jit(nopython=True)
def dissipationLoop(N,K2,uh,vh,wh,nu):
    sum = 0
    for i in range(np.shape(K2)[0]):
        for j in range(np.shape(K2)[1]):
            for k in range(np.shape(K2)[2]):
                sum += (K2[i,j,k]*((uh[i,j,k]+vh[i,j,k])+wh[i,j,k])/3)


    return np.real(2*nu*sum)



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
@jit(nopython=True)
def integralDissipation(curl_hat):
    dissipation = np.sum(np.abs(curl_hat) ** 2)
    return dissipation

def dissipationComputation(a, work, K,nu):
    """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
    curl_hat = work[(a, 0, False)]
    curl_hat = cross2a(curl_hat, K, a)
    #c[0] = ifft(curl_hat[0])
    #c[1] = ifft(curl_hat[1])
    #c[2] = ifft(curl_hat[2])
    dissipation = integralDissipation(curl_hat)
    return dissipation*nu

def dissipationComputation(b, work, K,nu):
    """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
    uh = fftn(b[0])/N_three
    vh = fftn(b[1])/N_three
    wh = fftn(b[2])/N_three
    tmp = array([uh,vh,wh])

    curl_hat = work[(tmp, 0, False)]
    curl_hat = cross2a(curl_hat, K, tmp)
    #c[0] = ifft(curl_hat[0])
    #c[1] = ifft(curl_hat[1])
    #c[2] = ifft(curl_hat[2])
    dissipation = integralEnergy(comm,curl_hat)
    return dissipation*nu

@jit(nopython=True,parallel=True)
def dissipationLoop(wave_numbers,nu,tke):
    sum = 0
    K2 = [x**2 for x in wave_numbers]
    for k in range(np.shape(K2)[0]):
        sum += (K2[k]*tke[k])
    return np.real(2*nu*sum)

@jit(nopython=True)
def BandEnergy(tke,kf):
    sum = 0
    for k in range(kf):
        sum += (tke[k])
    return sum


def integralEnergy(arg):
    #TODO make this function work in parallel?
    result = ((sum(abs(arg[...]) ** 2)))
    return result/N_three

def L2_norm(comm, arg,N_three):
    r"""Compute the L2-norm of real array a
    Computing \int abs(u)**2 dx
    """

    #result =
    result = comm.allreduce((sum(abs(arg[...]) ** 2)))
    return result/N_three


def compute_tke_spectrum(u, v, w, length, smooth):
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
    lx,ly,lz = length,length,length
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
    tke_spectrum = tke_spectrum / knorm

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
length=2*np.pi
xticks = np.logspace(0,2,7)
yticks = np.logspace(1,-13,5)
N=512
N_half = int(N/2)
kf = 8
nu = 1/1600
amount = 50
name = 'vel_files_iso/velocity_'+str(step)+'.npy'
plot = 'spectrum2'
counter =0
dissipationArray = np.zeros((amount))
stepjump = 120
timearray = np.arange(0,(amount*100),stepjump)/100
energyarrayKf = []
energyarrayKin = []

runLoop = True

if runLoop == True:
    kx = fftfreq(N, 1. / N)
    K = np.array(np.meshgrid(kx, kx, kx, indexing='ij'), dtype=int)
    K2 = np.sum(K * K, 0, dtype=int)
    #TODO load in one and one file from /vel_files, read [0][:,:,-1] and add to animation. Also make spectrum plots and viscous diffusion plots
    for i in range(amount):
        name = 'vel_files_iso/velocity_' + str(step) + '.npy'
        vec = np.load(name)
        print('Loaded nr: '+str(step),flush=True)
        if plot == 'plotVelocity':
            im = plt.imshow(vec[0][:,:,-1],cmap='jet', animated=True)
            ims.append([im])
        if plot == 'isoVelocity':
            im = plt.imshow(vec[0][:,-1,:], animated=True)
            plt.savefig('iso_images/velocity_' + str(step))
            plt.clf()
        if plot == 'spectrum1':
            fig = spectrum(N,vec[0],vec[1],vec[2])
            plt.savefig('spectrum_plots/spectrum_'+str(step))
            #im = plt.show()
            #ims.append([im])
        if plot == 'spectrum2':
            nyquist,k,tke = compute_tke_spectrum(vec[0],vec[1],vec[2],length,True)
            eps = dissipationLoop(k, nu, tke)
            kinBand = BandEnergy(tke, kf)
            kinTotal = BandEnergy(tke, k[-1])
            #energyarrayKin.append(kinBand)
            #plt.plot(timearray[0:len(energyarrayKin)] , energyarrayKin, 'r--')


            plt.loglog(k[1:N_half],tke[1:N_half],'g.','markerSize=2')
            plt.loglog(k[1:N_half],(k[1:N_half]**(-5/3))*(eps**(-2/3)),'r--')
            plt.yscale('log')
            plt.ylim(ymin=(1e-18), ymax=1e3)
            # plt.xticks(xticks)
            # plt.yticks(yticks)
            plt.xlabel('Wave number, $k$')
            plt.ylabel('Turbulent kinetic energy, $E(k)$')
            plt.legend(['$E(k)$,  t= %.2f'%(step/100), r'$\epsilon^{-2/3}k^{-5/3}$'],loc='lower left')

            plt.savefig('spectrum_plots/spectrum_'+str(step))
            plt.clf()
        if plot == 'dissipation':
            Nt = N**3
            uh = fftn(vec[0])/Nt
            vh = fftn(vec[1])/Nt
            wh = fftn(vec[2])/Nt
            u_hat = np.array([uh,vh,wh])
            dissipationArray[counter]= dissipationComputation(u_hat,work_array,K,nu)
            counter +=1



        step += stepjump
        print('Finished appending nr: '+str(step),flush=True)
    np.save('dissipation.npy',dissipationArray)
    plt.plot(timearray,dissipationArray)


    '''
    if plot =='plotVelocity':
        ani = animation.ArtistAnimation(fig, ims, interval=2, blit=False,repeat_delay=None)
        ani.save('spectrum.gif', writer='imagemagick')
    '''
else:
    dissipation = np.load('dissipation.npy')
    plt.plot(timearray,dissipation,'k-')
    plt.xlabel('Time (s)')
    plt.ylabel(r'Enstrophy, $\epsilon$  ($\frac{m^2}{s^2}$)')
    plt.savefig('dissipation')