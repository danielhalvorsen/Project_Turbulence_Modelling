import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
try:
    from tqdm import tqdm
except ImportError:
    pass
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
from mpi4py import MPI
from functools import partial
#from multiprocessing import Pool


class Interpolator():
    """ Interpolating the datasets velocity components using regular grid interpolator (linear)
    interpolation over a rectangular mesh.
    The memberfunction get_interpolators returns functions for the
    velocity components' interpolated value at arbitrary positions.

    Parameters
    ----------
    dataset : xarray_type
            Data structure containing the oceanographic data.
    X       : array_type
            Particle coordinates.
    t       : datetime64_type
            Time.
    ----------
    """

    def __init__(self, velocityField):
        self.dataset = velocityField
        self.L = 2*np.pi
        self.N = len(self.dataset[0])
        self.x_vec = np.arange(0, self.N, 1) * self.L / self.N
        self.y_vec = np.arange(0, self.N, 1) * self.L / self.N
        self.z_vec = np.arange(0, self.N, 1) * self.L / self.N

    def get_interpolators(self, X,t):
        # Add a buffer of cells around the extent of the particle cloud
        if t<100:
            buf = 3
        else:
            buf=0
        # Find extent of particle cloud in terms of indices
        imax = np.searchsorted(self.x_vec, np.amax(X[0, :])) + buf
        imin = np.searchsorted(self.x_vec, np.amin(X[0, :])) - buf
        jmax = np.searchsorted(self.y_vec, np.amax(X[1, :])) + buf
        jmin = np.searchsorted(self.y_vec, np.amin(X[1, :])) - buf
        kmax = np.searchsorted(self.z_vec, np.amax(X[2, :])) + buf
        kmin = np.searchsorted(self.z_vec, np.amin(X[2, :])) - buf
        # Take out subset of array, to pass to RectBivariateSpline
        # Transpose to get regular order of coordinates (x,y)
        # Fill NaN values (land cells) with 0, otherwise
        # interpolation won't work
        u = self.dataset[0, imin:imax, jmin:jmax,kmin:kmax]
        v = self.dataset[1, imin:imax, jmin:jmax,kmin:kmax]
        w = self.dataset[2, imin:imax, jmin:jmax, kmin:kmax]
        self.dataset=None
        xslice = self.x_vec[imin:imax]
        yslice = self.y_vec[jmin:jmax]
        zslice = self.z_vec[kmin:kmax]
        # RectBivariateSpline returns a function-like object,
        # which can be called to get value at arbitrary position
        fu = RegularGridInterpolator((xslice,yslice,zslice),u,method='linear',bounds_error=False,fill_value=None)
        del(u)
        fv = RegularGridInterpolator((xslice,yslice,zslice),v,method='linear',bounds_error=False,fill_value=None)
        del(v)
        fw = RegularGridInterpolator((xslice,yslice,zslice),w,method='linear',bounds_error=False,fill_value=None)
        del(w)
        return fu, fv, fw

    def __call__(self, X,t):
        #X = np.where(X > (L - ldx), X - L, X)
        #X = np.where(X < 0, X + (L-ldx), X)

        # get index of current time in dataset
        # get interpolating functions,
        # covering the extent of the particle
        fu, fv, fw = self.get_interpolators(X,t)
        # Evaluate velocity at position(x[:], y[:])
        X = X.transpose()


        vx = fu(X)
        del(fu)
        vy = fv(X)
        del(fv)
        vz = fw(X)
        del(fw)
        return np.array([vx, vy, vz])


def rk2(x, t, h, f):
    """ A second order Rung-Kutta method.
        The Explicit Trapezoid Method.

    Parameters:
    -----------
        x :    coordinates (as an array of vectors)
        h :    timestep
        f :    A function that returns the derivatives
    Returns:
        Next coordinates (as an array of vectors)
    -----------
    """

    # Note: t and h have actual time units.
    # For multiplying with h, we need to
    # convert to number of seconds:
    dt = h
    # "Slopes"
    k1 = f(x, t)
    k2 = f(x + k1 * dt, t + h)
    # Calculate next position
    x_ = x + dt * (k1 + k2) / 2
    return x_


def Euler(x, t, h, f):
    """ A first order Rung-Kutta method.
        The Explicit Euler method.

    Parameters:
    -----------
        x :    coordinates (as an array of vectors)
        h :    timestep
        f :    A function that returns the derivatives
    Returns:
        Next coordinates (as an array of vectors)
    -----------
    """

    # Note: t and h have actual time units.
    # For multiplying with h, we need to
    # convert to number of seconds:
    dt = h
    # Calculate next position
    #TODO find way to use this variable only once
    Np = np.shape(x)[1]
    x_ = x + dt * f(x, t)
    x_ += np.random.normal(loc=0,scale=np.sqrt(dt),size=(3,Np))*np.sqrt(2*0.0005)
    return x_


@jit(nopython=True,fastmath=True)
def periodicBC(X,L,ldx):
    X = np.where(X > (L - ldx), X - (L - ldx), X)
    X = np.where(X < 0, X + (L - ldx), X)
    return X
def plotScatter(fig_particle,ax_particle,i,X,rgbaTuple,pointSize,L,pointcolor1,pointcolor2,Np):
    ax_particle.scatter(X[i + 1][0, 0:int(Np/2)], X[i + 1][1, 0:int(Np/2)], X[i + 1][2, 0:int(Np/2)], s=pointSize, c=pointcolor1)
    ax_particle.scatter(X[i + 1][0, int(Np/2):], X[i + 1][1, int(Np/2):], X[i + 1][2, int(Np/2):], s=pointSize, c=pointcolor2)

    plt.xlim([0, L])
    plt.ylim([0, L])
    ax_particle.set_zlim(0, L)
    ax_particle.set_xlabel('x-axis')
    ax_particle.set_ylabel('y-axis')
    ax_particle.set_zlabel('z-axis')

    ax_particle.w_xaxis.set_pane_color(rgbaTuple)
    ax_particle.w_yaxis.set_pane_color(rgbaTuple)
    ax_particle.w_zaxis.set_pane_color(rgbaTuple)

    ax_particle.xaxis.set_major_formatter(
        tck.FuncFormatter(lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'))
    ax_particle.xaxis.set_major_locator(tck.MultipleLocator(base=np.pi))
    ax_particle.yaxis.set_major_formatter(
        tck.FuncFormatter(lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'))
    ax_particle.yaxis.set_major_locator(tck.MultipleLocator(base=np.pi))
    ax_particle.zaxis.set_major_formatter(
        tck.FuncFormatter(lambda val, pos: '{:.0g}$\pi$'.format(val / np.pi) if val != 0 else '0'))
    ax_particle.zaxis.set_major_locator(tck.MultipleLocator(base=np.pi))

    #plt.pause(0.05)
    plt.savefig('./Particle_plots/test_'+str(i),dpi=600)
    ax_particle.clear()

def particle_IC(Np,L,choice):
    if choice=='random two slots':
        Np_int = int(Np / 2)
        par_Pos_init = np.zeros((3, Np))
        par_Pos_init[0, 0:Np_int] = np.random.uniform(L / 2 - L / 3, L / 2 - L / 3.5, size=Np_int)
        par_Pos_init[1, 0:Np_int] = np.random.uniform(L / 2 - L / 3, L / 2 - L / 3.5, size=Np_int)
        par_Pos_init[2, 0:Np_int] = np.random.uniform(L / 2 - L / 3, L / 2 - L / 3.5, size=Np_int)

        par_Pos_init[0, Np_int:] = np.random.uniform(L / 2 + L / 3, L / 2 + L / 3.5, size=Np_int)
        par_Pos_init[1, Np_int:] = np.random.uniform(L / 2 + L / 3, L / 2 + L / 3.5, size=Np_int)
        par_Pos_init[2, Np_int:] = np.random.uniform(L / 2 + L / 3, L / 2 + L / 3.5, size=Np_int)
    if choice=='middlePoint':
        par_Pos_init = np.zeros((3, Np))
        par_Pos_init[0, :] = L/2
        par_Pos_init[1, :] = L/2
        par_Pos_init[2, :] = L/2
    return par_Pos_init

def trajectory(t0, Tmax, h, f, integrator,dynamicField,L,ldx,X0):
    """ Function to calculate trajectory of the particles.

    Parameters:
    -----------
        X0 :    A two dimensional array containing start positions
                (x0, y0) of each particle.
        t0 :    Initial time
        Tmax:   Final time
        h  :    Timestep
        f  :    Interpolator
        integrator:   The chosen integrator function

    Returns:
        A three dimensional array containing the positions of
        each particle at every timestep on the interval (t0, Tmax).
    -----------
    """
    if (dynamicField==False):
        Nt = int((Tmax - t0) / h)  # Number of datapoints
        X = np.zeros((Nt + 2, *X0.shape))
        X[0, :] = X0
        t = t0
        try:
            pbar = tqdm(total=Nt)
        except:
            pass
        for i in range(Nt + 1):
            # Adjust last timestep to match Tmax exactly
            h = min(h, Tmax - t)
            t += h
            X[i + 1, :] = integrator(X[i, :], t, h, f)
            X[i+1,:] = periodicBC(X[i+1,:], L, ldx)


            #plotScatter(fig_particle, ax_particle, i, X, rgbaTuple, pointSize, L, pointcolor1, pointcolor2, Np)
            try:
                pbar.update(1)
            except:
                pass
        return X
    if (dynamicField==True):
        t=t0 #This variable is not used since we use explicit Euler method atm.
        X_new = integrator(X0,t,h,f)
        X_new = periodicBC(X_new,L,ldx)
        return X_new
'''
if __name__=='__main__':
    velField = np.load('vel_files_iso/velocity_120.npy')
    f  = Interpolator(velField)

    # Set initial conditions (t0 and x0) and timestep
    # Note that h also has time units, for convenient
    # calculation of t + h.

    # setting X0 in a slightly roundabout manner for
    # compatibility with Np >= 1

    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    rank = comm.Get_rank()
    #let Np be a multiple of num_processes
    Np = num_processes*50
    N=64
    L = np.pi*2
    ldx = L / N
    par_Pos_init = particle_IC(Np,L)

    fig_particle = plt.figure()
    ax_particle = fig_particle.add_subplot(111, projection='3d')
    pointSize = 3.1
    pointcolor1 = 'r'
    pointcolor2 = 'm'
    rgbaTuple = (167/255, 201/255, 235/255)

    #TODO wont need Tmax unless we collect new velocity field every time step.
    h  = 0.01
    t0 = 0
    Tmax = 5
    N1_particle = int(Np/num_processes)
    timesteps = int(Tmax/h+2)

    split_coordinates = np.array_split(par_Pos_init, num_processes, axis=1)
    data = comm.scatter(split_coordinates, root=0)

    #X1_full = np.empty((timesteps,3,Np))
    X1 = np.empty((timesteps, 3, N1_particle))

    X1 = trajectory(t0, Tmax, h, f, Euler, False, L, ldx, data)
    X1_full = comm.gather(X1, root=0)
    if rank==0:

        X1_reshaped = np.concatenate(X1_full,axis=2)
        for i in range(int(Tmax/h+2)):
            plotScatter(fig_particle, ax_particle, i, X1_reshaped, rgbaTuple, pointSize, L, pointcolor1, pointcolor2, Np)
'''
