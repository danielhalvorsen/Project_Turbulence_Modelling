# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:14:22 2016

@author: Xin
"""
#solve 2-D incompressible NS equations using spectral method 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os.path
from numpy import *
from numpy.fft import fftfreq , fft , ifft, fft2, ifft2, fftshift, ifftshift
from mpi4py import MPI
from tqdm import tqdm
parent = os.path.abspath(os.path.join(os.path.dirname(__file__),'.'))
sys.path.append(parent)


#parameters
new=1;
Nstep=10000; #no. of steps
N=Nx=Ny=64; #grid size
t=0;
nu=5e-10; #viscosity
nu_hypo=2e-3; #hypo-viscosity
dt=5e-5; #time-step
dt_h=dt/2; #half-time step
ic_type=2 #1 for Taylor-Green init_cond; 2 for random init_cond
k_ic=1;  #initial wavenumber for Taylor green forcing
diag_out_step = 2500; #frequency of outputting diagnostics

#------------MPI setup---------
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
rank = comm.Get_rank()
#slab decomposition, split arrays in x direction in physical space, in ky direction in Fourier space
Np = int(N/num_processes)



#---------declare functions that will be used----------

#---------2D FFT and IFFT-----------
Uc_hat = empty(( N, Np) , dtype = complex )
Uc_hatT = empty((Np,N) , dtype = complex )
U_mpi = empty (( num_processes , Np , Np ) , dtype = complex )

#inverse FFT
def ifftn_mpi(fu, u):
    Uc_hat[:]=ifftshift(ifft(fftshift(fu) , axis=0))
    comm.Alltoall([Uc_hat, MPI.DOUBLE_COMPLEX], [U_mpi, MPI.DOUBLE_COMPLEX])
    Uc_hatT[:] = rollaxis(U_mpi, 1).reshape(Uc_hatT.shape)
    u[:]= ifftshift(ifft (fftshift(Uc_hatT), axis =1))
    return u
#FFT
def fftn_mpi(u, fu):
    Uc_hatT[:] = fftshift(fft(ifftshift(u), axis=1))
    U_mpi[:] = rollaxis(Uc_hatT.reshape(Np, num_processes, Np), 1)
    comm.Alltoall([U_mpi, MPI.DOUBLE_COMPLEX], [fu, MPI.DOUBLE_COMPLEX])
    fu[:]= fftshift(fft(ifftshift(fu), axis=0))
    return fu  
    

#initialize x,y kx, ky coordinate
def IC_coor(Nx, Ny, Np, dx, dy, rank, num_processes): 
    x=zeros((Np, Ny), dtype=float);
    y=zeros((Np, Ny), dtype=float);
    kx=zeros((Nx, Np), dtype=float);
    ky=zeros((Nx, Np), dtype=float);
    for j in range(Ny):
        x[0:Np,j]=range(Np);
        if num_processes == 1:
            x[0:Nx,j] =range(int(-Nx/2), int(Nx/2));
    #offset for mpi
    if num_processes != 1:
        x=x-(num_processes/2-rank)*Np
    x=x*dx;
    for i in range(Np):
	    y[i,0:Ny] =range(int(-Ny/2), int(Ny/2));
    y=y*dy;
	
    for j in range(Np):
	    kx[0:Nx,j]=range(int(-Nx/2), int(Nx/2));
    for i in range(Nx):
        ky[i,0:Np]=range(Np);
        if num_processes == 1:
            ky[i,0:Ny]=range(int(-Ny/2), int(Ny/2));
    #offset for mpi
    if num_processes != 1:
        ky=ky-(num_processes/2-rank)*Np
        
    k2=kx**2+ky**2;
    for i in range(Nx):
	    for j in range(Np):
	         if(k2[i,j] == 0):
	             k2[i,j]=1e-5; #so that I do not divide by 0 below when using projection operator	
    k2_exp=exp(-nu*(k2**5)*dt-nu_hypo*dt);    
    return x, y, kx, ky, k2, k2_exp
    
    
#---------Dealiasing function----
def delias(u_hat, v_hat, Nx, Np, k2):
    #use 1/3 rule to remove values of wavenumber >= Nx/3
    for i in range(Nx):
        for j in range(Np):
            if(sqrt(k2[i,j]) >= Nx/3.):
                u_hat[i,j]=0;
                v_hat[i,j]=0;
    #Projection operator on velocity fields to make them solenoidal-----
    tmp = (kx*u_hat + ky*v_hat)/k2;
    u_hat = u_hat - kx*tmp;
    v_hat = v_hat - ky*tmp;
    return u_hat, v_hat


#----Initialize Velocity in Fourier space-----------
def IC_condition(ic_type, k_ic, kx, ky, Nx, Np):
    #taylor green vorticity field
    u_hat = zeros((Nx, Np), dtype=complex);
    v_hat = zeros((Nx, Np), dtype=complex);
    if (new==1 and ic_type==1):
        for iss in [-1, 1]:
            for jss in [-1, 1]:
                for i in range(Nx):
                    for j in range(Np):
                        if(int(kx[i,j])==iss*k_ic and int(ky[i,j])==jss*k_ic):
                            u_hat[i,j] = -1j*iss;
                            v_hat[i,j] = -1j*(-jss);
        #Set total energy to 1
        u_hat=0.5*u_hat;
        v_hat=0.5*v_hat;
    #generate random velocity field
    elif (new==1 and ic_type==2):
        u=random.rand(Np,Ny)
        v=random.rand(Np,Ny)
        u_hat=fftn_mpi(u, u_hat)
        v_hat=fftn_mpi(v, v_hat)
    return u_hat, v_hat

#------output function----
#this function output vorticty contour
def output(omega, x, y, Nx, Ny, rank, time):
   #collect values to root 
        omega_all=comm.gather(omega, root=0)
        x_all=comm.gather(x, root=0)
        y_all=comm.gather(y, root=0)
        if rank==0:
            #reshape the ''list''
            omega_all=asarray(omega_all).reshape(Nx, Ny)
            x_all=asarray(x_all).reshape(Nx, Ny)
            y_all=asarray(y_all).reshape(Nx, Ny)
            plt.contourf(x_all, y_all, omega_all,cmap='jet')
            delimiter = ''
            title = delimiter.join(['vorticity contour, time=', str(time)])
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(title)
            plt.show()
            filename = delimiter.join(['vorticity@t=', str(time),'.png'])
            plt.savefig(filename, format='png')

#--------------finish declaration of functions-------




#-----------GRID setup-----------
Lx=2*pi;
Ly=2*pi;
dx=Lx/Nx;
dy=Ly/Ny;

#obtain x, y, kx, ky
x, y, kx, ky, k2, k2_exp=IC_coor(Nx, Ny, Np, dx, dy, rank, num_processes)
	
#----Initialize Variables-------(hat denotes variables in Fourier space,)
#velocity
u_hat=zeros((Nx, Np), dtype=complex);
v_hat=zeros((Nx, Np), dtype=complex);
#Vorticity
omega_hat=zeros((Nx, Np), dtype=complex);
#Nonlinear term
NLxhat=zeros((Nx, Np), dtype=complex);
NLyhat=zeros((Nx, Np), dtype=complex);
#variables in physical space
u=zeros((Np, Ny), dtype=float);
v=zeros((Np, Ny), dtype=float);
omega=zeros((Np, Ny), dtype=float);

#generate initial velocity field
u_hat, v_hat=IC_condition(ic_type, k_ic, kx, ky, Nx, Np)
      
#------Dealiasing------------------------------------------------
u_hat, v_hat=delias(u_hat, v_hat, Nx, Np, k2)
#
#------Storing variables for later use in time integration--------
u_hat_t0 = u_hat;
v_hat_t0 = v_hat;
#

step = 1
pbar = tqdm(total=int(Nstep))


#----Main Loop-----------
for istep in range(Nstep+1):
    if rank==0:
        wt=MPI.Wtime()
    #------Dealiasing
    u_hat, v_hat=delias(u_hat, v_hat, Nx, Np, k2)
    #Calculate Vorticity
    omega_hat = 1j*(kx*v_hat - ky*u_hat);
    #fields in x-space
    u=ifftn_mpi(u_hat,u)
    v=ifftn_mpi(v_hat,v)
    omega=ifftn_mpi(omega_hat,omega)
    
    #Fields in Fourier Space
    u_hat=fftn_mpi(u, u_hat)
    v_hat=fftn_mpi(v, v_hat)
    omega_hat=fftn_mpi(omega, omega_hat)
   
    #Calculate non-linear term in x-space
    NLx =  v*omega;
    NLy = -u*omega;
    
    
    #move non-linear term back to Fourier k-space
    NLxhat=fftn_mpi(NLx, NLxhat)
    NLyhat=fftn_mpi(NLy, NLyhat)
    
    #------Dealiasing------------------------------------------------
    u_hat, v_hat=delias(u_hat, v_hat, Nx, Np, k2)
    
    #Integrate in time
    #---Euler for 1/2-step-----------
    if(istep==0):
          u_hat = u_hat + dt_h*(NLxhat -nu*(k2**5)*u_hat -nu_hypo*(k2**(-0))**u_hat);
          v_hat = v_hat + dt_h*(NLyhat -nu*(k2**5)*v_hat -nu_hypo*(k2**(-0))**v_hat);
          oNLxhat = NLxhat;
          oNLyhat = NLyhat;
    #---Midpoint time-integration----
    elif(istep==1):
          u_hat = u_hat_t0 + dt*(NLxhat -nu*(k2**5)*u_hat -nu_hypo*(k2**(-0))*u_hat);
          v_hat = v_hat_t0 + dt*(NLyhat -nu*(k2**5)*v_hat -nu_hypo*(k2**(-0))*v_hat);

    #---Adam-Bashforth integration---
    else:
          u_hat = u_hat + dt*(1.5*NLxhat - 0.5*oNLxhat*k2_exp);
          v_hat = v_hat + dt*(1.5*NLyhat - 0.5*oNLyhat*k2_exp);
          u_hat = u_hat*k2_exp;
          v_hat = v_hat*k2_exp;
          u_hat = u_hat;
          v_hat = v_hat;

          oNLxhat=NLxhat;
          oNLyhat=NLyhat;
    #output vorticity contour
    if(istep%diag_out_step==0):
        output(omega, x, y, Nx, Ny, rank, t)
        if rank==0:
            print('simulation time')
            print(MPI.Wtime()-wt)

    t=t+dt;
    step+=1
    pbar.update(1)
