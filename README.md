# Project - Turbulence Modelling #
### 3D DNS solver for the Navier-Stokes equations using a spectral method ###

* Study of turbulent diffusion




<p align="center">
<img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/3D/iso256Nice.gif" width="350"/><img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/3D/256spectrum_fixedaxis.gif" width="350"/>
</p>
3D-isotropic turbulence, N=256, Re=1600. See [1] for initialization and [2] for a section
on forcing the lowest wavenumbers (k<=kf=8) to maintain a constant turbulent
kinetic energy. Parameters related to the initial condition are set such that the kinetic energy matches Taylor-Green initial conditions.

---
---

<img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/3D/512_70_xy.gif" width="280"/> <img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/3D/512_70_xz.gif" width="280"/><img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/3D/512_70_yz.gif" width="280"/>
3D-isotropic turbulence using Re=1600. Taylor Green Vortex. N=512. The planes presented are respectively the xy-, xz- and yz-plane with last accessible index in corresponding axis. The code ran on the Idun cluster using 128 cores for 24 hours. 

<p align="center">
<img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/3D/spectrum512_43.gif" width="350"/><img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/3D/dissipation.png" width="350"/>
</p>
Computed energy spectrum E(k) for the time range t in [0,42]. Right image shows enstrophy distribution with a peak dissipation at t=9 seconds.

<img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/3D/TG3D_64.gif" width="425"/> <img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/3D/TG3D_128.gif" width="425"/>
3D-isotropic turbulence using Re=1.6M. Taylor Green Vortex. Left: N=64, Right: N=128.
---
---


### 2D DNS solver for the Navier-Stokes equations using a spectral method ###
## Note: timescales on animations not fixed. ##
 
<img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/2D/nice.gif" width="425"/> <img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/2D/fieldspread.gif" width="425"/>
Left animation: 2D- vorticity field. Re=1600, N=256. Smaller vortices give energy to larger vortices. Right animation: 2D Advection-Diffusion equation solved with an initial concentration distribution. Velocity distribution from the left vorticity field. Diffusion constant set to be 0.0008

<img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/2D/nice2.gif" width="425"/> <img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/2D/fieldspread2.gif" width="425"/>
Left animation: 2D- vorticity field. Re=1600, N=256. Smaller vortices give energy to larger vortices. Right animation: 2D Advection-Diffusion equation solved with an initial concentration distribution. Velocity distribution from the left vorticity field. Diffusion constant set to be 0.0008
---
---
### References
[1] R. S. Rogallo, "Numerical experiments in homogeneous turbulence,"
NASA TM 81315 (1981)

[2] A. G. Lamorgese and D. A. Caughey and S. B. Pope, "Direct numerical simulation
of homogeneous turbulence with hyperviscosity", Physics of Fluids, 17, 1, 015106,
2005, (https://doi.org/10.1063/1.1833415)