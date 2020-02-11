import numpy as np
import time
from mpi4py import MPI

import matplotlib
from pathlib import Path

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

try:
    from tqdm import tqdm
except ImportError:
    pass

#TODO make this path dynamic to fit different platforms, mac/linux etc
path = Path('/home/danieloh/PycharmProjects/Project_Turbulence_Modelling/2D/Advection')

# Set the colormap
plt.rcParams['image.cmap'] = 'BrBG'

# Basic parameters
N = 64
dt = 1e-3
tend = 30

D = 0.0008  # Diffusion constant
L = 2 * np.pi  # 2pi = one period
dx = L / N
dy = dx  # equidistant
dz = dx  # equidistant
dx2 = dx ** 2
dy2 = dy ** 2
dz2 = dz ** 2
r = dt * D / (dx ** 2)
assert (0 < 2 * r <= 0.5), ('N too high. Reduce time step to uphold stability')
timesteps = int(np.ceil(tend / dt))
load_every = 0.01*timesteps
breaker = load_every
image_interval = timesteps * 0.05  # Write frequency for png files
field_store_counter=0


filenames_u = ['datafiles/u/u_vel_t_0.npy']
filenames_v = ['datafiles/v/v_vel_t_0.npy']

for i in range(1,int(timesteps/load_every)+1):
    filenames_u.append('datafiles/u/u_vel_t_'+str(i)+'.npy')
    filenames_v.append('datafiles/v/v_vel_t_'+str(i)+'.npy')


print('finished loading file names')

x = np.arange(0, N, 1) * L / N
y = np.arange(0, N, 1) * L / N
[X, Y] = np.meshgrid(x, y)

fig, axs = plt.subplots(2)
fig.suptitle('Title here')

# For stability, this is the largest interval possible
# for the size of the time-step:
# dt = dx2*dy2 / ( 2*D*(dx2+dy2) )


# MPI globals
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Up/down neighbouring MPI ranks
up = rank - 1
if up < 0:
    up = size - 1
down = rank + 1
if down > size - 1:
    down = 0


def evolve(sol_new, sol, u, v, D, dt, dx2, dy2):
    """Explicit time evolution.
       u:            new temperature field
       u_previous:   previous field
       a:            diffusion constant
       dt:           time step
       dx2:          grid spacing squared, i.e. dx^2
       dy2:            -- "" --          , i.e. dy^2"""
    # LEFT boundary

    sol_new[1:-1, 0] = sol[1:-1, 0] \
                       + (dt * u[1:-1, 0]) / (2 * dx) * (sol[:-2, 0] - sol[2:, 0]) \
                       + (dt * v[1:-1, 0]) / (2 * dx) * (sol[1:-1, -1] - sol[1:-1, 1]) \
                       + r * (sol[2:, 0] - 2 * sol[1:-1, 0] + sol[:-2, 0]) \
                       + r * (sol[1:-1, 1] - 2 * sol[1:-1, 0] + sol[1:-1, -1])
    # INNER points
    sol_new[1:-1, 1:-1] = sol[1:-1, 1:-1] \
                          + (dt * u[1:-1, 1:-1]) / (2 * dx) * (sol[:-2, 1:-1] - sol[2:, 1:-1]) \
                          + (dt * v[1:-1, 1:-1]) / (2 * dx) * (sol[1:-1, :-2] - sol[1:-1, 2:]) \
                          + r * (sol[2:, 1:-1] - 2 * sol[1:-1, 1:-1] + sol[:-2, 1:-1]) \
                          + r * (sol[1:-1, 2:] - 2 * sol[1:-1, 1:-1] + sol[1:-1, :-2])
    # RIGHT boundary
    sol_new[1:-1, -1] = sol[1:-1, -1] \
                        + (dt * u[1:-1, -1]) / (2 * dx) * (sol[:-2, -1] - sol[2:, -1]) \
                        + (dt * v[1:-1, -1]) / (2 * dx) * (sol[1:-1, -2] - sol[1:-1, 0]) \
                        + r * (sol[2:, -1] - 2 * sol[1:-1, -1] + sol[:-2, -1]) \
                        + r * (sol[1:-1, 0] - 2 * sol[1:-1, -1] + sol[1:-1, -2])

    # Update next time step
    sol[:] = sol_new[:]
    # sol[:] = sol[:]/(np.sum(sol[:])) #cheat with mass conservation. Assume uniform loss over each cell





def init_fields(X, Y):
    # Read the initial temperature field from file
    # field = np.loadtxt(filename)
    # field0 = field.copy()  # Array for field of previous time step
    pos = np.dstack((X, Y))
    mu = np.array([2, 3])
    cov = np.array([[.05, .010], [.010, .05]])
    rv = multivariate_normal(mu, cov)
    S = rv.pdf(pos)
    field = S.copy() / (np.sum(S))
    field0 = field.copy()

    # A = 0.1
    # omega = 1
    # epsilon = 0.25
    # u = -np.pi*A*np.sin(np.pi*(1)*X)*np.cos(np.pi*Y)
    # v = -np.pi*A*np.cos(np.pi*(1)*X)*np.sin(np.pi*Y)*(1-2*epsilon)

    u = np.sin(1 * X) * np.cos(1 * Y) * 0.1
    v = np.cos(1 * X) * np.cos(1 * Y) * 0.1
    # u = np.random.rand(N, N)*1
    # v = np.random.rand(N, N)*1
    # u = np.ones((N,N))*1
    # v = np.ones((N,N))*1
    cu = dt * np.max(u) / dx
    cv = dt * np.max(v) / dx
    assert (((cu ** 2) / r) + ((cv ** 2) / r) <= 2), ('dt might be too high or diffusion constant might be too low')

    return field, field0, u, v


def write_field(field, step):
    plt.gca().clear()
    plt.imshow(field, cmap='jet')
    plt.axis('off')
    plt.savefig('heat_{0:03d}.png'.format(step))


def exchange(field):
    # send down, receive from up
    sbuf = field[-2, :]
    rbuf = field[0, :]
    comm.Sendrecv(sbuf, dest=down, recvbuf=rbuf, source=up)
    # send up, receive from down
    sbuf = field[1, :]
    rbuf = field[-1, :]
    comm.Sendrecv(sbuf, dest=up, recvbuf=rbuf, source=down)


def iterate(field, local_field, local_field0, timesteps, image_interval,field_store_counter):
    step = 1
    try:
        pbar = tqdm(total=int(timesteps))
    except:
        pass
    counter=0
    indexnr = 1
    for t in range(0, timesteps ):
        if (t == 0):
            u_init = np.load(filenames_u[0])
            v_init = np.load(filenames_v[0])
            shape = u_init.shape
            dtype = u_init.dtype
            comm.bcast(shape, 0)  # broadcast dimensions
            comm.bcast(dtype, 0)  # broadcast data type
            n = int(shape[0] / size)  # number of rows for each MPI task
            m = shape[1]  # number of columns in the field
            buff = np.zeros((n, m), dtype)

            comm.Scatter(u_init, buff, 0)  # scatter the data
            local_u = np.zeros((n + 2, m), dtype)  # need two ghost rows!
            local_u[1:-1, :] = buff  # copy data to non-ghost rows

            comm.Scatter(v_init, buff, 0)  # scatter the data
            local_v = np.zeros((n + 2, m), dtype)  # need two ghost rows!
            local_v[1:-1, :] = buff  # copy data to non-ghost rows

            exchange(local_field0)
            exchange(local_u)
            exchange(local_v)
            evolve(local_field, local_field0, local_u, local_v, D, dt, dx2, dy2)

        if (t != 0):
            if (t%breaker==1):
                u = np.load(filenames_u[indexnr])
                v = np.load(filenames_v[indexnr])
                indexnr +=1
                shape = u[0].shape
                dtype = u[0].dtype
                comm.bcast(shape, 0)  # broadcast dimensions
                comm.bcast(dtype, 0)  # broadcast data type
                n = int(shape[0] / size)  # number of rows for each MPI task
                m = shape[1]  # number of columns in the field
                buff = np.zeros((n, m), dtype)

            comm.Scatter(u[int(counter%breaker)], buff, 0)  # scatter the data
            local_u = np.zeros((n + 2, m), dtype)  # need two ghost rows!
            local_u[1:-1, :] = buff  # copy data to non-ghost rows

            comm.Scatter(v[int(counter%breaker)], buff, 0)  # scatter the data
            local_v = np.zeros((n + 2, m), dtype)  # need two ghost rows!
            local_v[1:-1, :] = buff  # copy data to non-ghost rows



            exchange(local_field0)
            exchange(local_u)
            exchange(local_v)
            evolve(local_field, local_field0, local_u, local_v, D, dt, dx2, dy2)
            step += 1
            counter +=1
            try:
                pbar.update(1)
            except:
                pass

        if(t%breaker==0):
            comm.Gather(local_field[1:-1, :], field, root=0)
            if (rank==0):
                np.save('datafiles/concentrations/field_' + str(round(t)), field)

        '''
        if (t % image_interval == 0 and t!=0):
            comm.Gather(local_field[1:-1, :], field, root=0)
            comm.Gather(local_u[1:-1, :], u[int(counter%breaker)], root=0)
            comm.Gather(local_v[1:-1, :], v[int(counter%breaker)], root=0)
            if rank == 0:
                # write_field(field, m)
                # plt.imshow(field.T,cmap='jet')
                axs[0].imshow(field, cmap='jet')  # ,vmax=1,vmin=0)
                axs[1].imshow((v[int(counter%breaker)]**2+u[int(counter%breaker)]**2), cmap='jet')
                # plt.show()
                plt.pause(0.05)

        '''

def main():
    # Read and scatter the initial temperature field
    if rank == 0:
        field, field0, u, v = init_fields(X, Y)
        shape = field.shape
        dtype = field.dtype
        comm.bcast(shape, 0)  # broadcast dimensions
        comm.bcast(dtype, 0)  # broadcast data type
    else:
        field = None
        u = None
        v = None
        shape = comm.bcast(None, 0)
        dtype = comm.bcast(None, 0)
    if shape[0] % size:
        raise ValueError('Number of rows in the field (' \
                         + str(shape[0]) + ') needs to be divisible by the number ' \
                         + 'of MPI tasks (' + str(size) + ').')
    n = int(shape[0] / size)  # number of rows for each MPI task
    m = shape[1]  # number of columns in the field
    buff = np.zeros((n, m), dtype)
    comm.Scatter(field, buff, 0)  # scatter the data
    local_field = np.zeros((n + 2, m), dtype)  # need two ghost rows!
    local_field[1:-1, :] = buff  # copy data to non-ghost rows
    local_field0 = np.zeros_like(local_field)  # array for previous time step
    local_field0[:] = local_field[:]

    t0 = time.time()
    print('starting iterations')
    iterate(field, local_field, local_field0, timesteps, image_interval,field_store_counter)
    print('end iterations')
    t1 = time.time()

if __name__ == '__main__':
    main()
