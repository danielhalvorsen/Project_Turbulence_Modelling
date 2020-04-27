import matplotlib.pyplot as plt
import numpy as np
plt.style.use('bmh')

run='2'

VarX = np.load('./Variance_files/run'+run+'/VarX_list.npy')
VarY = np.load('./Variance_files/run'+run+'/VarY_list.npy')
VarZ = np.load('./Variance_files/run'+run+'/VarZ_list.npy')

combined_variance = []

for i in range(len(VarX)):
    combined_variance.append((VarX[i]+VarY[i]+VarZ[i])/2)


time = np.arange(0,80,0.01)
tsteps = len(time)+1
b = 2*np.pi
a = 0
exact_var = (1/12)*((b-a)**2)


linewidth_variance = 0.5
linewidth_polyfit = 1

idxa_fick = 1
idxb_fick = 90

idxc_total = 610
idxd_total = 4000

localidx_turb_a = 230
localidx_turb_b = 518

listToPlot = VarZ.copy()
liststring = 'z'

m1,c1 = np.polyfit(np.log(time[idxa_fick:idxb_fick]),np.log(listToPlot[idxa_fick:idxb_fick]),1)
log_fit1 = m1*np.log(time[idxa_fick:idxb_fick])+c1

m2,c2 = np.polyfit(np.log(time[idxc_total:idxd_total]),np.log(listToPlot[idxc_total:idxd_total]),1)
log_fit2 = m2*np.log(time[idxc_total:idxd_total])+c2

m3,c3 = np.polyfit(np.log(time[localidx_turb_a:localidx_turb_b]),np.log(listToPlot[localidx_turb_a:localidx_turb_b]),1)
log_fit3 = m3*np.log(time[localidx_turb_a:localidx_turb_b])+c3


#plt.loglog(t[idxc:idxd],tpower1[idxc:idxd],'b--')
plt.loglog(time[idxa_fick:idxb_fick],np.exp(log_fit1),color='k',linestyle=(0,(3,1,1,1,1,1)),linewidth=linewidth_polyfit,label=r'$\mathrm{Fickian-diffusion},\; t^{%.2f}$'%(m1))
plt.loglog(time[idxc_total:idxd_total],np.exp(log_fit2),color='k',linestyle='dashdot',linewidth=linewidth_polyfit,label='$\mathrm{Turbulent-diffusion},\; t^{%.2f}$'%(m2))
#plt.loglog(time[localidx_turb_a:localidx_turb_b],np.exp(log_fit3),color='k',linestyle=(0,(1,1)),linewidth=linewidth_polyfit,label=r'$\mathrm{Richardson-scaling},\; t^{%.2f}}$'%(m3))




print('exact var..: ',exact_var)
plt.plot(time,listToPlot[:-1],'b',linewidth=linewidth_variance,label=r'$\sigma^{2}\mathrm{,\;'+liststring+'-component}$')
#plt.plot(time,VarY[:-1],'g',linewidth=linewidth_variance,label='Variance in PD y-component')
#plt.plot(time,VarZ[:-1],'m',linewidth=linewidth_variance,label='Variance in PD z-component')
#plt.plot(time,combined_variance[:-1],'m',linewidth=linewidth_variance,label='Combined variance')

plt.plot(time,exact_var*np.ones(len(time)),'r--',linewidth=1,label=r'$\mathrm{Uniform \;distribution}$')


plt.yscale(value="log")
plt.xscale(value="log")
plt.xlim((0.01,80))
plt.ylim((1e-5,10))
plt.xlabel('$\mathrm{Time \;(s)}$')
plt.ylabel('$\sigma^{2} \; \mathrm{(m^{2})}$')
plt.legend()
plt.savefig('./Variance_files/Plots/variance_'+liststring+'_PD_'+run+'.png',dpi=1000)
plt.show()
