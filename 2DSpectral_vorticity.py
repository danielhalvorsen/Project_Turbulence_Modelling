# Solver of 2D Navier Stokes equation on streamfunction-vorticity formulation.

import numpy as np
import pickle
from numpy.fft import fftfreq, fft, ifft, irfft2, rfft2
import random
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from tqdm import tqdm

nu = 1e-4

L = np.pi
N = int(128)
N2 = int(N ** 2)
dx = 2 * L / N
mesh = np.mgrid[:N, :N] * 2 * L / N
X = mesh[0]
Y = mesh[1]

kx = fftfreq(N, 1. / N)
sacrifice = np.arange(N / 4 + 2, N / 4 * 3 + 1).astype(int)
kx[sacrifice] = 0
ky = kx.copy()
K = np.array(np.meshgrid(kx, ky), dtype=int)
Kx = K[0]
Ky = K[1]
K2 = np.sum(K * K, 0, dtype=int)
K2_inv = 1 / np.where(K2 == 0, 1, K2).astype(float)
Dx = 1j * Kx * K2_inv
Dy = 1j * Ky * K2_inv

t0 = 0
t_end = 4
dt = 0.1
t = np.linspace(t0,t_end,np.ceil(t_end/dt))

#np.random.seed(seed=MT19937)
omega = np.zeros([N,N])
omega_hat = np.fft.fft2(omega)
omega_hat[1,5] = np.random.rand(1)+1j*np.random.rand(1)
#omega_hat[2,5] = np.random.rand(1)+1j*np.random.rand(1)
#omega_hat[5,2] = np.random.rand(1)+1j*np.random.rand(1)
#omega_hat[10,10] = np.random.rand(1)+1j*np.random.rand(1)
omega_hat[2,2] = np.random.rand(1)+1j*np.random.rand(1)
omega_hat[4,1] = np.random.rand(1)+1j*np.random.rand(1)
omega = np.real(np.fft.ifft2(omega_hat))
omega = omega/np.max(omega)
omega_vector = np.reshape(omega, N2, 1)


def Rhs(t, omega_vector):
    omega = np.reshape(omega_vector, ([N, N]))
    omega_hat = np.fft.fft2(omega)
    omx = np.real(np.fft.ifft2(1j * Kx * omega_hat))
    omy = np.real(np.fft.ifft2(1j * Ky * omega_hat))
    u = np.real(np.fft.ifft2(Dy * omega_hat))
    v = np.real(np.fft.ifft2(-Dx * omega_hat))
    rhs = np.real(np.fft.ifft2(-nu * K2 * omega_hat) - u * omx - v * omy)
    Rhs = np.reshape(rhs, N2, 1)
    return Rhs


#'''
#numsteps = np.ceil(t_end/dt)
#step = 1
#pbar = tqdm(total = int(t_end/dt))
#while step<=numsteps:
solve = integrate.solve_ivp(Rhs, [0, t_end], omega_vector, method='RK45',rtol=1e-8,atol=1e-8)
#omega_vector = solve.y[:,1]
omega = np.reshape(solve.y[:,-1], ([N, N]))

with open('solve_matrix'+'.pkl','wb') as f:
    pickle.dump([solve],f)

#step+=1
#pbar.update(1)
#pbar.close()


#print(np.shape(solve.y[:,0]))
#plt.contourf(omega.T,levels=500)
#print(solve)
#plt.show()
#'''