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
Nstep=5000; #no. of steps
N=Nx=Ny=256; #grid size
t=0;
nu=5e-10; #viscosity
nu_hypo=2e-3; #hypo-viscosity
dt=5e-7; #time-step
dt_h=dt/2; #half-time step
ic_type=2 #1 for Taylor-Green init_cond; 2 for random init_cond
k_ic=1;  #initial wavenumber for Taylor green forcing
diag_out_step = 1000; #frequency of outputting diagnostics

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
def delias(Vxhat, Vyhat, Nx, Np, k2):
    #use 1/3 rule to remove values of wavenumber >= Nx/3
    for i in range(Nx):
        for j in range(Np):
            if(sqrt(k2[i,j]) >= Nx/3.):
                Vxhat[i,j]=0;
                Vyhat[i,j]=0;
    #Projection operator on velocity fields to make them solenoidal-----
    tmp = (kx*Vxhat + ky*Vyhat)/k2;
    Vxhat = Vxhat - kx*tmp;
    Vyhat = Vyhat - ky*tmp;
    return Vxhat, Vyhat


#----Initialize Velocity in Fourier space-----------
def IC_condition(ic_type, k_ic, kx, ky, Nx, Np):
    #taylor green vorticity field
    Vxhat = zeros((Nx, Np), dtype=complex);
    Vyhat = zeros((Nx, Np), dtype=complex);
    if (new==1 and ic_type==1):
        for iss in [-1, 1]:
            for jss in [-1, 1]:
                for i in range(Nx):
                    for j in range(Np):
                        if(int(kx[i,j])==iss*k_ic and int(ky[i,j])==jss*k_ic):
                            Vxhat[i,j] = -1j*iss;
                            Vyhat[i,j] = -1j*(-jss);               
        #Set total energy to 1
        Vxhat=0.5*Vxhat;     
        Vyhat=0.5*Vyhat;
    #generate random velocity field
    elif (new==1 and ic_type==2):
        Vx=random.rand(Np,Ny)
        Vy=random.rand(Np,Ny)
        Vxhat=fftn_mpi(Vx, Vxhat)
        Vyhat=fftn_mpi(Vy, Vyhat)     
    return Vxhat, Vyhat

#------output function----
#this function output vorticty contour
def output(Wz, x, y, Nx, Ny, rank, time):
   #collect values to root 
        Wz_all=comm.gather(Wz, root=0)
        x_all=comm.gather(x, root=0)
        y_all=comm.gather(y, root=0)
        if rank==0:
            #reshape the ''list''
            Wz_all=asarray(Wz_all).reshape(Nx, Ny)
            x_all=asarray(x_all).reshape(Nx, Ny)
            y_all=asarray(y_all).reshape(Nx, Ny)
            plt.contourf(x_all, y_all, Wz_all,cmap='jet')
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
Vxhat=zeros((Nx, Np), dtype=complex);
Vyhat=zeros((Nx, Np), dtype=complex);
#Vorticity
Wzhat=zeros((Nx, Np), dtype=complex);
#Nonlinear term
NLxhat=zeros((Nx, Np), dtype=complex);
NLyhat=zeros((Nx, Np), dtype=complex);
#variables in physical space
Vx=zeros((Np, Ny), dtype=float);
Vy=zeros((Np, Ny), dtype=float);
Wz=zeros((Np, Ny), dtype=float);

#generate initial velocity field
Vxhat, Vyhat=IC_condition(ic_type, k_ic, kx, ky, Nx, Np)
      
#------Dealiasing------------------------------------------------
Vxhat, Vyhat=delias(Vxhat, Vyhat, Nx, Np, k2)
#
#------Storing variables for later use in time integration--------
Vxhat_t0 = Vxhat;
Vyhat_t0 = Vyhat;
#

step = 1
pbar = tqdm(total=int(Nstep))


#----Main Loop-----------
for istep in range(Nstep+1):
    if rank==0:
        wt=MPI.Wtime()
    #------Dealiasing
    Vxhat, Vyhat=delias(Vxhat, Vyhat, Nx, Np, k2)
    #Calculate Vorticity
    Wzhat = 1j*(kx*Vyhat - ky*Vxhat);
    #fields in x-space
    Vx=ifftn_mpi(Vxhat,Vx)
    Vy=ifftn_mpi(Vyhat,Vy)
    Wz=ifftn_mpi(Wzhat,Wz)
    
    #Fields in Fourier Space
    Vxhat=fftn_mpi(Vx, Vxhat)
    Vyhat=fftn_mpi(Vy, Vyhat)
    Wzhat=fftn_mpi(Wz, Wzhat)
   
    #Calculate non-linear term in x-space
    NLx =  Vy*Wz;
    NLy = -Vx*Wz;
    
    
    #move non-linear term back to Fourier k-space
    NLxhat=fftn_mpi(NLx, NLxhat)
    NLyhat=fftn_mpi(NLy, NLyhat)
    
    #------Dealiasing------------------------------------------------
    Vxhat, Vyhat=delias(Vxhat, Vyhat, Nx, Np, k2)
    
    #Integrate in time
    #---Euler for 1/2-step-----------
    if(istep==0):
          Vxhat = Vxhat + dt_h*(NLxhat -nu*(k2**5)*Vxhat -nu_hypo*(k2**(-0))**Vxhat);
          Vyhat = Vyhat + dt_h*(NLyhat -nu*(k2**5)*Vyhat -nu_hypo*(k2**(-0))**Vyhat);
          oNLxhat = NLxhat;
          oNLyhat = NLyhat;
    #---Midpoint time-integration----
    elif(istep==1):
          Vxhat = Vxhat_t0 + dt*(NLxhat -nu*(k2**5)*Vxhat -nu_hypo*(k2**(-0))*Vxhat);
          Vyhat = Vyhat_t0 + dt*(NLyhat -nu*(k2**5)*Vyhat -nu_hypo*(k2**(-0))*Vyhat);          
    #---Adam-Bashforth integration---
    else:
          Vxhat = Vxhat + dt*(1.5*NLxhat - 0.5*oNLxhat*k2_exp);
          Vyhat = Vyhat + dt*(1.5*NLyhat - 0.5*oNLyhat*k2_exp);
          Vxhat = Vxhat*k2_exp;
          Vyhat = Vyhat*k2_exp;
          Vxhat = Vxhat;
          Vyhat = Vyhat;

          oNLxhat=NLxhat;
          oNLyhat=NLyhat;
    #output vorticity contour
    if(istep%diag_out_step==0):
        output(Wz, x, y, Nx, Ny, rank, t)
        if rank==0:
            print('simulation time')
            print(MPI.Wtime()-wt)

    t=t+dt;
    step+=1
    pbar.update(1)
