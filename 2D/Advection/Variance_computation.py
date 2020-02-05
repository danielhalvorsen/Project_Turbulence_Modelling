import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats as sp
from scipy import integrate
from skimage import measure
from matplotlib import rc
from basic_units import radians, degrees, cos
from radians_plot import *
import powerlaw as pl

plt.style.use('bmh')

tend = 100
dt = 1e-3
timesteps = int(np.ceil(tend / dt))

N = 256
L = 2 * np.pi
dx = L / N
x = np.arange(0, N, 1) * L / N
y = np.arange(0, N, 1) * L / N
[X, Y] = np.meshgrid(x, y)

con_x = []
con_y = []

var_x = []
var_y = []

field_mean_x = []
field_mean_Y = []

fig, axs = plt.subplots(1)

field_mean_x = []
field_mean_y = []


def closest(lst, K):
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]


def intargument(field, N, x,mean_or_central,mean_x,mean_y):

    con_x = field.sum(axis=0)
    con_y = field.sum(axis=1)
    mean_npx = np.mean(con_x)
    mean_npy = np.mean(con_y)
    close_value_x = closest(con_x, mean_npx)
    close_value_y = closest(con_y, mean_npy)

    if mean_x =='str' and mean_y =='str':
        meanvalue_x = x[np.where(con_x == close_value_x)][0]
        meanvalue_y = y[np.where(con_y == close_value_y)][0]
    else:
        meanvalue_x = float(mean_x)
        meanvalue_y = float(mean_y)

    argumentx = []
    argumenty = []
    for i in range(N):
        argumentx.append(((x[i] - meanvalue_x*mean_or_central) ** 2) * con_x[i])
        argumenty.append(((y[i] - meanvalue_y*mean_or_central) ** 2) * con_y[i])
    return argumentx, argumenty,meanvalue_x,meanvalue_y

def first_moment(field, N, x):

    con_x = field.sum(axis=0)
    con_y = field.sum(axis=1)
    argumentfirstx = []
    argumentfirsty = []
    for i in range(N):
        argumentfirstx.append(((x[i])) * con_x[i])
        argumentfirsty.append(((y[i])) * con_y[i])
    return argumentfirstx, argumentfirsty

def second_moment(field, N, x,firstx,firsty):

    con_x = field.sum(axis=0)
    con_y = field.sum(axis=1)
    argumentx = []
    argumenty = []
    for i in range(N):
        argumentx.append(((x[i]-firstx)**2) * con_x[i])
        argumenty.append(((y[i]-firsty)**2) * con_y[i])
    return argumentx, argumenty



intlist = []
mean_or_central=0
flipbool = False
meanvalue_x = 'str'
meanvalue_y = 'str'

for i in range(0, 100):
    #print(i)
    flipbool = False
    field = np.load('datafiles/con_1/field_' + str(i * 1000) + '.npy')

    [argumentx, argumenty,meanvalue_x,meanvalue_y] = intargument(field, N, x,mean_or_central,0,0)
    half_central_integral_x = integrate.simps(argumentx, x=x, dx=dx) * 0.5
    half_central_integral_y = integrate.simps(argumenty, x=x, dx=dx) * 0.5

    idx_x = 1
    while flipbool==False:
        lengthX = idx_x
        xlist = x[0:idx_x]
        running_central_integral_x = integrate.simps(argumentx[0:idx_x], x=xlist, dx=dx)
        idx_x +=1
        if running_central_integral_x >= half_central_integral_x:
            flipbool = True
    idx_y = 1
    flipbool = False
    while flipbool==False:
        lengthX = idx_y
        xlist = x[0:idx_y]
        running_central_integral_y = integrate.simps(argumenty[0:idx_y], x=xlist, dx=dx)
        idx_y +=1
        if running_central_integral_y >= half_central_integral_y:
            flipbool = True


    #[firstmomentx_arg,firstmomenty_arg]=first_moment(field,N,x)
    #firstmomentx = integrate.simps(firstmomentx_arg,x=x,dx=dx)
    #firstmomenty = integrate.simps(firstmomenty_arg,x=x,dx=dx)
    #[argumentx,argumenty]=second_moment(field,N,x,firstmomentx,firstmomenty)
    mean_or_central=0
    [argumentx, argumenty, meanvalue_x, meanvalue_y] = intargument(field, N, x, mean_or_central, x[idx_x], x[idx_y])
    intlist.append(integrate.simps(argumenty, x=x, dx=dx))
    con_x.append(field.sum(axis=0))
    con_y.append(field.sum(axis=1))
    var_x.append(np.var(field, axis=0))
    var_y.append(np.var(field, axis=1))
    field_mean_x.append((np.mean(var_x[i])))
    field_mean_y.append((np.mean(var_y[i])))
   # print(meanvalue_y)
    '''
    if (i%9==0):
        print(i)
        plt.contourf(X,Y,field,levels=15,xunits=radians, yunits=radians,cmap='jet')
        plt.xlim([0,2*np.pi])
        plt.ylim([0, 2 * np.pi])
        #plt.xlabel('x-direction (m)')
        #plt.ylabel('y-direction (m)')
        ax = plt.gca()
        ax.set_xlabel('x-direction (m)')
        ax.set_ylabel('y-direction (m)')

        ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        plt.colorbar()
        plt.show()
    '''
    '''
    axs[0].imshow(field, cmap='jet')
    axs[1].plot(x,con_x[i])
    axs[1].plot([meanvalue_x],[0],'r*')

    axs[2].plot(x,con_y[i])
    axs[2].plot([meanvalue_y],[0],'r*')
    axs[3].plot(intlist)
    plt.pause(0.03)
    axs[0].axes.clear()
    axs[1].axes.clear()
    axs[2].axes.clear()
    axs[3].axes.clear()
    '''


idxa = 2
idxb = 7
idxc = 9
idxd = 20


t = np.arange(0,100,1)
tpower1 = 0.18*t**(-0.2)
tpower2 = 0.026*t**(1)
#tpower = new_list = [n-0 for n in tpower]


m1,c1 = np.polyfit(np.log(t[idxa:idxb]),np.log(intlist[idxa:idxb]),1)
log_fit1 = m1*np.log(t[idxa:idxb])+c1

m2,c2 = np.polyfit(np.log(t[idxc:idxd]),np.log(intlist[idxc:idxd]),1)
log_fit2 = m2*np.log(t[idxc:idxd])+c2


#plt.loglog(t[idxc:idxd],tpower1[idxc:idxd],'b--')
plt.loglog(t[idxa:idxb],np.exp(log_fit1),'b--')
plt.loglog(t[idxc:idxd],np.exp(log_fit2),'r--')

#plt.loglog(t[idxa:idxb],tpower2[idxa:idxb],'r--')
plt.loglog(intlist,'k-')
axs.set_xscale('log')
plt.xlabel('$\mathrm{time \;(s)}$')
plt.ylabel('$\\sigma_{y}^{2} \; \mathrm{(m^2)}$')

plt.legend(['powerlaw $\propto t^{%.2f}$'%(m1),'powerlaw $\propto t^{%.2f}$'%(m2),'SM'])
plt.show()
