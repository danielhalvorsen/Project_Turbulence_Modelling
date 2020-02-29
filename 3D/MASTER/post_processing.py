import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.fft import fftfreq, fft, ifft, irfft2, fftn,fftshift,rfft,irfft
import time
import matplotlib
matplotlib.use('TkAgg')







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




#fig = plt.figure()
ims = []
step=0
N=512
amount = 32
name = 'vel_files/velocity_'+str(step)+'.npy'
plot = 'spectrum'

#TODO load in one and one file from /vel_files, read [0][:,:,-1] and add to animation. Also make spectrum plots and viscous diffusion plots
for i in range(amount):
    name = 'vel_files/velocity_' + str(step) + '.npy'
    vec = np.load(name)
    print('Loaded nr: '+str(step),flush=True)
    if plot == 'plotVelocity':
        im = plt.imshow(vec[0][:,:,-1],cmap='jet', animated=True)
        ims.append([im])
    if plot == 'spectrum':
        fig = spectrum(N,vec[0],vec[1],vec[2])
        plt.savefig('spectrum_plots/spectrum_'+str(step))
        #im = plt.show()
        #ims.append([im])
    step += 140
    print('Finished appending nr: '+str(step),flush=True)

'''
if plot =='spectrum':
    ani = animation.ArtistAnimation(fig2, ims, interval=1, blit=False,repeat_delay=None)
    ani.save('spectrum.gif', writer='imagemagick')

'''