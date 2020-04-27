import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')


time=0

step=160
for i in range(55):
    time = 0
    time += step*i
    IC = 'isotropic'
    N=512
    N_half = int(N/2)

    k = np.load('./spectral_files/'+IC+'/wave_numbers_'+str(time)+'.npy')
    TKE = np.load('./spectral_files/'+IC+'/TKE'+str(time)+'.npy')


    plt.loglog(k[2:N_half],TKE[2:N_half],'g-',markerSize=2)
    plt.loglog(k[2:N_half],(k[2:N_half]**(-5/3)),'r--')
    plt.yscale(value='log')
    plt.xscale(value='log',basex=2)
    plt.ylim(ymin=(1e-18), ymax=1e3)
    plt.xlabel('$\mathrm{k}$')
    plt.ylabel('$\mathrm{E(k)}$')
    plt.legend(['$\mathrm{E(k)}$,  $\mathrm{t= %.2f}$'%(time/100), r'$\mathrm{k^{-5/3}}$'],loc='lower left')
    plt.savefig('./spectrum_plots/'+IC+'/TKE_'+str(time)+'.png',dpi=1000)
    plt.cla()