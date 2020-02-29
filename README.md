# Project - Turbulence Modelling #
### 3D DNS solver for the Navier-Stokes equations using a spectral method ###

* Study of turbulent diffusion






<img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/512_70_xy.gif" width="280"/> <img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/512_70_xz.gif" width="280"/><img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/512_70_yz.gif" width="280"/>
3D-isotropic turbulence using Re=1600. Taylor Green Vortex. N=512. The planes presented are respectively the xy-, xz- and yz-plane with last accessible index in corresponding axis. The code ran on the Idun cluster using 128 cores for 24 hours. 

<p align="center">
<img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/spectrumgif_512_1600_70.gif" width="350"/>
Computed energy spectrum <img src="/tex/0b73f1dfec99f4698e9399208969619d.svg?invert_in_darkmode&sanitize=true" align=middle width=34.94297234999999pt height=24.65753399999998pt/> for the time range <img src="/tex/25a967376d35e3217b572b3aaf7fc6e0.svg?invert_in_darkmode&sanitize=true" align=middle width=21.46113749999999pt height=20.221802699999984pt/> [0,42].
</p>

<img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/TG3D_64.gif" width="425"/> <img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/TG3D_128.gif" width="425"/>
3D-isotropic turbulence using Re=1.6M. Taylor Green Vortex. Left: N=64, Right: N=128.




### 2D DNS solver for the Navier-Stokes equations using a spectral method ###
## Note: timescales on animations not fixed. ##
 
<img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/nice.gif" width="425"/> <img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/fieldspread.gif" width="425"/>
Left animation: 2D- vorticity field. Re=1600, N=256. Smaller vortices give energy to larger vortices. Right animation: 2D Advection-Diffusion equation solved with an initial concentration distribution. Velocity distribution from the left vorticity field. Diffusion constant set to be 0.0008
 
 

<img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/nice2.gif" width="425"/> <img src="https://github.com/danielhalvorsen/Project_Turbulence_Modelling/blob/master/animation_folder/fieldspread2.gif" width="425"/>
Left animation: 2D- vorticity field. Re=1600, N=256. Smaller vortices give energy to larger vortices. Right animation: 2D Advection-Diffusion equation solved with an initial concentration distribution. Velocity distribution from the left vorticity field. Diffusion constant set to be 0.0008