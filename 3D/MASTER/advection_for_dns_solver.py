import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

def up_or_down(rank,size):
    up = rank - 1
    if up < 0:
        # this applies for rank==0, then the up-rank is responsible for the local field at bottom of the data array.
        up = size - 1
    down = rank + 1
    if down > size - 1:
        # If this is the last rank, the down rank should be the one responsible for the local field at the top of the data array, i.e. down=0
        down = 0
    return up,down

def init_field_advection(X, L):
    mean = L/2
    variance = 0.01
    field = np.exp(-((X[0] - mean) ** 2 + (X[1] - mean) ** 2 + (
            X[2] - mean) ** 2) / variance)  # Solution matrix of advection-diffusion equation
    field = field / np.sum(field)
    return field


def cell_computation_inner_evolve(Nx, Nk, U, S_old, S_new, dx, dy, dz, dx2, dy2, dz2,D,dt):
    # nx = (i + 1) % N
    # ny = (j + 1) % N
    # nz = (k + 1) % N

    s_l = slice(0, Nx - 2)
    s_m = slice(1, Nx - 1)
    s_r = slice(2, Nx)

    s_l_k = slice(0, Nk - 2)
    s_m_k = slice(1, Nk - 1)
    s_r_k = slice(2, Nk)

    con_x = U[0][s_m, s_m, s_m_k] * (S_old[s_r, s_m, s_m_k] - S_old[s_l, s_m, s_m_k]) / (2 * (dx))
    con_y = U[1][s_m, s_m, s_m_k] * (S_old[s_m, s_r, s_m_k] - S_old[s_m, s_l, s_m_k]) / (2 * (dy))
    con_z = U[2][s_m, s_m, s_m_k] * (S_old[s_m, s_m, s_r_k] - S_old[s_m, s_m, s_l_k]) / (2 * (dz))
    diff_x = D * (S_old[s_r, s_m, s_m_k] - 2 * S_old[s_m, s_m, s_m_k] + S_old[s_l, s_m, s_m_k]) / (dx2)
    diff_y = D * (S_old[s_m, s_r, s_m_k] - 2 * S_old[s_m, s_m, s_m_k] + S_old[s_m, s_l, s_m_k]) / (dy2)
    diff_z = D * (S_old[s_m, s_m, s_r_k] - 2 * S_old[s_m, s_m, s_m_k] + S_old[s_m, s_m, s_l_k]) / (dz2)
    residual = -con_x - con_y - con_z + diff_x + diff_y + diff_z
    S_old_lax = S_old[s_m, s_m, s_m_k].copy()
    # S_old_lax = (1 / 6) * ((S_old[s_l, s_m, s_m] + S_old[s_r, s_m, s_m]) + (
    #        S_old[s_m, s_l, s_m] + S_old[s_m, s_r, s_m]) + (S_old[s_m, s_m, s_l] + S_old[s_m, s_m, s_r]))
    S_new[s_m, s_m, s_m_k] = S_old_lax + dt * residual
    # return S_new[s_m, s_m, s_m]


def cell_computation_front_evolve(Nx, Nk, U, S_old, S_new, dx, dy, dz, dx2, dy2, dz2,D,dt):
    # nx = (i + 1) % N
    # ny = (j + 1) % N
    # nz = (k + 1) % N

    s_l = slice(0, Nx - 2)
    s_m = slice(1, Nx - 1)
    s_r = slice(2, Nx)

    s_l_k = slice(0, Nk - 2)
    s_m_k = slice(1, Nk - 1)
    s_r_k = slice(2, Nk)

    ## left points on front plane
    con_x = U[0][0, 0, s_m_k] * (S_old[1, 0, s_m_k] - S_old[-1, 0, s_m_k]) / (2 * (dx))
    con_y = U[1][0, 0, s_m_k] * (S_old[0, 1, s_m_k] - S_old[0, -1, s_m_k]) / (2 * (dy))
    con_z = U[2][0, 0, s_m_k] * (S_old[0, 0, s_r_k] - S_old[0, 0, s_l_k]) / (2 * (dz))
    diff_x = D * (S_old[1, 0, s_m_k] - 2 * S_old[0, 0, s_m_k] + S_old[-1, 0, s_m_k]) / (dx2)
    diff_y = D * (S_old[0, 1, s_m_k] - 2 * S_old[0, 0, s_m_k] + S_old[0, -1, s_m_k]) / (dy2)
    diff_z = D * (S_old[0, 0, s_r_k] - 2 * S_old[0, 0, s_m_k] + S_old[0, 0, s_l_k]) / (dz2)
    residual = -con_x - con_y - con_z + diff_x + diff_y + diff_z
    # S_old_lax = (1 / 6) * ((S_old[-1, 0, s_m_k] + S_old[1, 0, s_m_k]) + (
    #        S_old[0, -1, s_m_k] + S_old[0, 1, s_m_k]) + (S_old[0, 0, s_l_k] + S_old[0, 0, s_r_k]))
    S_old_lax = S_old[0, 0, s_m_k]
    S_new[0, 0, s_m_k] = S_old_lax + dt * residual

    ## inner points on front plane
    con_x = U[0][s_m, 0, s_m_k] * (S_old[s_r, 0, s_m_k] - S_old[s_l, 0, s_m_k]) / (2 * (dx))
    con_y = U[1][s_m, 0, s_m_k] * (S_old[s_m, 1, s_m_k] - S_old[s_m, -1, s_m_k]) / (2 * (dy))
    con_z = U[2][s_m, 0, s_m_k] * (S_old[s_m, 0, s_r_k] - S_old[s_m, 0, s_l_k]) / (2 * (dz))
    diff_x = D * (S_old[s_r, 0, s_m_k] - 2 * S_old[s_m, 0, s_m_k] + S_old[s_l, 0, s_m_k]) / (dx2)
    diff_y = D * (S_old[s_m, 1, s_m_k] - 2 * S_old[s_m, 0, s_m_k] + S_old[s_m, -1, s_m_k]) / (dy2)
    diff_z = D * (S_old[s_m, 0, s_r_k] - 2 * S_old[s_m, 0, s_m_k] + S_old[s_m, 0, s_l_k]) / (dz2)
    residual = -con_x - con_y - con_z + diff_x + diff_y + diff_z
    # S_old_lax = (1 / 6) * ((S_old[s_l, 0, s_m_k] + S_old[s_r, 0, s_m_k]) + (
    #        S_old[s_m, -1, s_m_k] + S_old[s_m, 1, s_m_k]) + (S_old[s_m, 0, s_l_k] + S_old[s_m, 0, s_r_k]))
    S_old_lax = S_old[s_m, 0, s_m_k]
    S_new[s_m, 0, s_m_k] = S_old_lax + dt * residual

    ## right points on front plane
    con_x = U[0][-1, 0, s_m_k] * (S_old[0, 0, s_m_k] - S_old[-2, 0, s_m_k]) / (2 * (dx))
    con_y = U[1][-1, 0, s_m_k] * (S_old[-1, 1, s_m_k] - S_old[-1, -1, s_m_k]) / (2 * (dy))
    con_z = U[2][-1, 0, s_m_k] * (S_old[-1, 0, s_r_k] - S_old[-1, 0, s_l_k]) / (2 * (dz))
    diff_x = D * (S_old[0, 0, s_m_k] - 2 * S_old[-1, 0, s_m_k] + S_old[0, 0, s_m_k]) / (dx2)
    diff_y = D * (S_old[-1, 1, s_m_k] - 2 * S_old[-1, 0, s_m_k] + S_old[-1, -1, s_m_k]) / (dy2)
    diff_z = D * (S_old[-1, 0, s_r_k] - 2 * S_old[-1, 0, s_m_k] + S_old[-1, 0, s_l_k]) / (dz2)
    residual = -con_x - con_y - con_z + diff_x + diff_y + diff_z
    # S_old_lax = (1 / 6) * ((S_old[-2, 0, s_m_k] + S_old[0, 0, s_m_k]) + (
    #        S_old[-1, -1, s_m_k] + S_old[-1, 1, s_m_k]) + (S_old[-1, 0, s_l_k] + S_old[-1, 0, s_r_k]))
    S_old_lax = S_old[-1, 0, s_m_k]
    S_new[-1, 0, s_m_k] = S_old_lax + dt * residual
    # return S_new[:,0,:]


def cell_computation_back_evolve(Nx, Nk, U, S_old, S_new, dx, dy, dz, dx2, dy2, dz2,D,dt):
    # nx = (i + 1) % N
    # ny = (j + 1) % N
    # nz = (k + 1) % N

    s_l = slice(0, Nx - 2)
    s_m = slice(1, Nx - 1)
    s_r = slice(2, Nx)

    s_l_k = slice(0, Nk - 2)
    s_m_k = slice(1, Nk - 1)
    s_r_k = slice(2, Nk)

    ## left points on back plane
    con_x = U[0][0, -1, s_m_k] * (S_old[1, -1, s_m_k] - S_old[-1, -1, s_m_k]) / (2 * (dx))
    con_y = U[1][0, -1, s_m_k] * (S_old[0, 0, s_m_k] - S_old[0, -2, s_m_k]) / (2 * (dy))
    con_z = U[2][0, -1, s_m_k] * (S_old[0, -1, s_r_k] - S_old[0, -1, s_l_k]) / (2 * (dz))
    diff_x = D * (S_old[1, -1, s_m_k] - 2 * S_old[0, -1, s_m_k] + S_old[-1, 0, s_m_k]) / (dx2)
    diff_y = D * (S_old[0, 0, s_m_k] - 2 * S_old[0, -1, s_m_k] + S_old[0, -1, s_m_k]) / (dy2)
    diff_z = D * (S_old[0, -1, s_r_k] - 2 * S_old[0, -1, s_m_k] + S_old[0, 0, s_l_k]) / (dz2)
    residual = -con_x - con_y - con_z + diff_x + diff_y + diff_z
    # S_old_lax = (1 / 6) * ((S_old[-1, -1, s_m_k] + S_old[1, -1, s_m_k]) + (
    #        S_old[0, -2, s_m_k] + S_old[0, 0, s_m_k]) + (S_old[0, -1, s_l_k] + S_old[0, -1, s_r_k]))
    S_old_lax = S_old[0, -1, s_m_k]
    S_new[0, -1, s_m_k] = S_old_lax + dt * residual

    ## inner points on back plane
    con_x = U[0][s_m, -1, s_m_k] * (S_old[s_r, -1, s_m_k] - S_old[s_l, -1, s_m_k]) / (2 * (dx))
    con_y = U[1][s_m, -1, s_m_k] * (S_old[s_m, 0, s_m_k] - S_old[s_m, -2, s_m_k]) / (2 * (dy))
    con_z = U[2][s_m, -1, s_m_k] * (S_old[s_m, -1, s_r_k] - S_old[s_m, -1, s_l_k]) / (2 * (dz))
    diff_x = D * (S_old[s_r, -1, s_m_k] - 2 * S_old[s_m, -1, s_m_k] + S_old[s_l, -1, s_m_k]) / (dx2)
    diff_y = D * (S_old[s_m, 0, s_m_k] - 2 * S_old[s_m, -1, s_m_k] + S_old[s_m, -2, s_m_k]) / (dy2)
    diff_z = D * (S_old[s_m, -1, s_r_k] - 2 * S_old[s_m, -1, s_m_k] + S_old[s_m, -1, s_l_k]) / (dz2)
    residual = -con_x - con_y - con_z + diff_x + diff_y + diff_z
    # S_old_lax = (1 / 6) * ((S_old[s_l, -1, s_m_k] + S_old[s_r, -1, s_m_k]) + (
    #        S_old[s_m, -2, s_m_k] + S_old[s_m, 0, s_m_k]) + (S_old[s_m, -1, s_l_k] + S_old[s_m, -1, s_r_k]))
    S_old_lax = S_old[s_m, -1, s_m_k]
    S_new[s_m, -1, s_m_k] = S_old_lax + dt * residual

    ## right points on back plane
    con_x = U[0][-1, -1, s_m_k] * (S_old[0, -1, s_m_k] - S_old[-2, -1, s_m_k]) / (2 * (dx))
    con_y = U[1][-1, -1, s_m_k] * (S_old[-1, 0, s_m_k] - S_old[-1, -2, s_m_k]) / (2 * (dy))
    con_z = U[2][-1, -1, s_m_k] * (S_old[-1, -1, s_r_k] - S_old[-1, -1, s_l_k]) / (2 * (dz))
    diff_x = D * (S_old[0, -1, s_m_k] - 2 * S_old[-1, -1, s_m_k] + S_old[0, -1, s_m_k]) / (dx2)
    diff_y = D * (S_old[-1, 0, s_m_k] - 2 * S_old[-1, -1, s_m_k] + S_old[-1, -2, s_m_k]) / (dy2)
    diff_z = D * (S_old[-1, -1, s_r_k] - 2 * S_old[-1, -1, s_m_k] + S_old[-1, -1, s_l_k]) / (dz2)
    residual = -con_x - con_y - con_z + diff_x + diff_y + diff_z
    # S_old_lax = (1 / 6) * ((S_old[-2, -1, s_m_k] + S_old[0, -1, s_m_k]) + (
    #        S_old[-1, -2, s_m_k] + S_old[-1, 0, s_m_k]) + (S_old[-1, -1, s_l_k] + S_old[-1, -1, s_r_k]))
    S_old_lax = S_old[-1, -1, s_m_k]
    S_new[-1, -1, s_m_k] = S_old_lax + dt * residual
    # return S_new[:,-1,:]


def cell_computation_right_evolve(Nx, Nk, U, S_old, S_new, dx, dy, dz, dx2, dy2, dz2,D,dt):
    # nx = (i + 1) % N
    # ny = (j + 1) % N
    # nz = (k + 1) % N

    s_l = slice(0, Nx - 2)
    s_m = slice(1, Nx - 1)
    s_r = slice(2, Nx)

    s_l_k = slice(0, Nk - 2)
    s_m_k = slice(1, Nk - 1)
    s_r_k = slice(2, Nk)

    # Inner points on right-plane
    con_x = U[0][-1, s_m, s_m_k] * (S_old[0, s_m, s_m_k] - S_old[-2, s_m, s_m_k]) / (2 * (dx))
    con_y = U[1][-1, s_m, s_m_k] * (S_old[-1, s_r, s_m_k] - S_old[-1, s_l, s_m_k]) / (2 * (dy))
    con_z = U[2][-1, s_m, s_m_k] * (S_old[-1, s_m, s_r_k] - S_old[-1, s_m, s_l_k]) / (2 * (dz))
    diff_x = D * (S_old[0, s_m, s_m_k] - 2 * S_old[-1, s_m, s_m_k] + S_old[-2, s_m, s_m_k]) / (dx2)
    diff_y = D * (S_old[-1, s_r, s_m_k] - 2 * S_old[-1, s_m, s_m_k] + S_old[-1, s_l, s_m_k]) / (dy2)
    diff_z = D * (S_old[-1, s_m, s_r_k] - 2 * S_old[-1, s_m, s_m_k] + S_old[-1, s_m, s_l_k]) / (dz2)
    residual = -con_x - con_y - con_z + diff_x + diff_y + diff_z
    # S_old_lax = (1 / 6) * ((S_old[-2, s_m, s_m_k] + S_old[0, s_m, s_m_k]) + (
    #        S_old[-1, s_l, s_m_k] + S_old[-1, s_r, s_m_k]) + (S_old[-1, s_m, s_l_k] + S_old[-1, s_m, s_r_k]))
    S_old_lax = S_old[-1, s_m, s_m_k]
    S_new[-1, s_m, s_m_k] = S_old_lax + dt * residual

    # return S_new[-1, :, :]


def cell_computation_left_evolve(Nx, Nk, U, S_old, S_new, dx, dy, dz, dx2, dy2, dz2,D,dt):
    # nx = (i + 1) % N
    # ny = (j + 1) % N
    # nz = (k + 1) % N

    s_l = slice(0, Nx - 2)
    s_m = slice(1, Nx - 1)
    s_r = slice(2, Nx)

    s_l_k = slice(0, Nk - 2)
    s_m_k = slice(1, Nk - 1)
    s_r_k = slice(2, Nk)

    # Inner points on left-plane
    con_x = U[0][0, s_m, s_m_k] * (S_old[1, s_m, s_m_k] - S_old[-1, s_m, s_m_k]) / (2 * (dx))
    con_y = U[1][0, s_m, s_m_k] * (S_old[0, s_r, s_m_k] - S_old[0, s_l, s_m_k]) / (2 * (dy))
    con_z = U[2][0, s_m, s_m_k] * (S_old[0, s_m, s_r_k] - S_old[0, s_m, s_l_k]) / (2 * (dz))
    diff_x = D * (S_old[1, s_m, s_m_k] - 2 * S_old[0, s_m, s_m_k] + S_old[-1, s_m, s_m_k]) / (dx2)
    diff_y = D * (S_old[0, s_r, s_m_k] - 2 * S_old[0, s_m, s_m_k] + S_old[0, s_l, s_m_k]) / (dy2)
    diff_z = D * (S_old[0, s_m, s_r_k] - 2 * S_old[0, s_m, s_m_k] + S_old[0, s_m, s_l_k]) / (dz2)
    residual = -con_x - con_y - con_z + diff_x + diff_y + diff_z
    # S_old_lax = (1 / 6) * ((S_old[-1, s_m, s_m_k] + S_old[1, s_m, s_m_k]) + (
    #        S_old[0, s_l, s_m_k] + S_old[0, s_r, s_m_k]) + (S_old[0, s_m, s_l_k] + S_old[0, s_m, s_r_k]))
    S_old_lax = S_old[0, s_m, s_m_k]
    S_new[0, s_m, s_m_k] = S_old_lax + dt * residual

    # return S_new[0, :, :]


def evolve(Nx, Nk, U_local, S_old_local, S_new_local,dx,dy,dz,dx2,dy2,dz2,D,dt):
    # INNER
    cell_computation_inner_evolve(Nx, Nk, U_local, S_old_local, S_new_local, dx, dy, dz, dx2, dy2, dz2,D,dt)
    # Left BC
    cell_computation_left_evolve(Nx, Nk, U_local, S_old_local, S_new_local, dx, dy, dz, dx2, dy2, dz2,D,dt)
    # Right BC
    cell_computation_right_evolve(Nx, Nk, U_local, S_old_local, S_new_local, dx, dy, dz, dx2, dy2, dz2,D,dt)
    # Front Bc
    cell_computation_front_evolve(Nx, Nk, U_local, S_old_local, S_new_local, dx, dy, dz, dx2, dy2, dz2,D,dt)
    # Back Bc
    cell_computation_back_evolve(Nx, Nk, U_local, S_old_local, S_new_local, dx, dy, dz, dx2, dy2, dz2,D,dt)

    # note: [1:-1] indicates the inner points. indexing [-1] gives last element, but 1:-1 yields all but first and last.

    # S_old = S_new.copy()
    S_old_local[:, :, :] = [*S_new_local]
    # S_old_local[:, :, :] = np.abs([*S_new_local] / np.sum([*S_new_local]))


def exchange(field,rank,size,comm):
    # print('shape of field into exchange: ',np.shape(field))
    # print('I am rank ',rank,'. Up for me is rank ',up,' and down is rank ',down)
    # send down, receive from up
    up,down=up_or_down(rank,size)
    sbuf = np.ascontiguousarray(field[:, :, -2])
    rbuf = np.ascontiguousarray(field[:, :, 0])
    comm.Sendrecv(sbuf, dest=down, recvbuf=rbuf, source=up)
    field[:, :, 0] = rbuf
    # send up, receive from down
    sbuf = np.ascontiguousarray(field[:, :, 1])
    rbuf = np.ascontiguousarray(field[:, :, -1])
    comm.Sendrecv(sbuf, dest=up, recvbuf=rbuf, source=down)
    field[:, :, -1] = rbuf


def iterate(t,save_field_step, U, local_field, local_field0,dx,dy,dz,dx2,dy2,dz2,D,dt,rank,size,comm):
    # TODO make sure the velocity fields are taken from the DNS solver
    if rank==0:
        U_shape = U[0].shape  # shape of data array (dataframe)
        n = U_shape[0]  # number of rows for each MPI task
        m = U_shape[1]  # number of columns in the field
        o = int(U_shape[2] / size)
        sendbuf_null = U[0]
        sendbuf_one = U[1]
        sendbuf_two = U[2]


        #print('Shape of buffer in rank0: ', np.shape(buff))
    else:
        U_shape = None
        sendbuf_null=None
        sendbuf_one=None
        sendbuf_two=None
    U_shape = comm.bcast(U_shape,root=0)
    n = U_shape[0]  # number of rows for each MPI task
    m = U_shape[1]  # number of columns in the field
    o = int(U_shape[2] / size)
    #buff = comm.bcast(buff,root=0)
    '''
    buff = np.zeros((n, m, o), dtype=np.float32)
    comm.Scatter(sendbuf_null,buff, root=0)  # scatter the data
    local_u = np.zeros((n, m, o + 2), dtype=np.float32)  # need two ghost rows!
    local_u[:, :, 1:-1] = buff  # copy data to non-ghost rows

    comm.Scatter(sendbuf_one, buff, root=0)  # scatter the data
    local_v = np.zeros((n, m, o + 2), dtype=np.float32)  # need two ghost rows!
    local_v[:, :, 1:-1] = buff  # copy data to non-ghost rows

    comm.Scatter(sendbuf_two, buff, root=0)  # scatter the data
    local_w = np.zeros((n, m, o + 2), dtype=np.float32)  # need two ghost rows!
    local_w[:, :, 1:-1] = buff  # copy data to non-ghost rows
    '''

    if rank == 0:
        u_split = np.dsplit(sendbuf_null, size)
    else:
        u_split = None
    buff = comm.scatter(u_split, root=0)  # scatter the data
    local_u = np.zeros((n, m, o + 2), dtype=np.float32)  # need two ghost rows!
    local_u[:, :, 1:-1] = buff  # copy data to non-ghost rows
################
    if rank == 0:
        v_split = np.dsplit(sendbuf_one, size)
    else:
        v_split = None
    buff = comm.scatter(v_split, root=0)  # scatter the data
    local_v = np.zeros((n, m, o + 2), dtype=np.float32)  # need two ghost rows!
    local_v[:, :, 1:-1] = buff  # copy data to non-ghost rows
###########
    if rank == 0:
        w_split = np.dsplit(sendbuf_two, size)
    else:
        w_split = None
    buff = comm.scatter(w_split, root=0)  # scatter the data
    local_w = np.zeros((n, m, o + 2), dtype=np.float32)  # need two ghost rows!
    local_w[:, :, 1:-1] = buff  # copy data to non-ghost rows



    local_U = np.array([local_u, local_v, local_w])


    del(buff)
    exchange(local_field0,rank,size,comm)
    exchange(local_u,rank,size,comm)
    exchange(local_v,rank,size,comm)
    exchange(local_w,rank,size,comm)
    evolve(m, o + 2, local_U, local_field0, local_field,dx,dy,dz,dx2,dy2,dz2,D,dt)

    if t % save_field_step == 0:
        field_gathered = comm.gather(np.ascontiguousarray(np.abs(local_field[:, :, 1:-1])), root=0)
        if rank == 0:
            field_gathered = np.dstack(field_gathered)
            np.save('advection_concentration_field/concentration_field_t'+str(t)+'.npy',field_gathered)
            print('field step '+str(t)+' is saved...',flush=True)
            del field_gathered
    return local_field,local_field0

def advection_setup(N,L,rank,size,comm):
    # Read and scatter the initial temperature field
    if rank == 0:
        x_vec = np.arange(0, N, 1) * L / N
        X = np.meshgrid(x_vec, x_vec, x_vec)
        field= init_field_advection(X, L)
        del(X)
        del(x_vec)
        field_shape = field.shape
        field_dtype = field.dtype

    else:
        field = None
        field_shape = None
    field_shape = comm.bcast(field_shape,root=0)
    if field_shape[2] % size:
        raise ValueError('Number of rows in the field (' \
                         + str(field_shape[2]) + ') needs to be divisible by the number ' \
                         + 'of MPI tasks (' + str(size) + ').')
    n = field_shape[0]  # number of rows for each MPI task
    m = field_shape[1]  # number of columns in the field
    o = int(field_shape[2] / size)
    #buff = np.ascontiguousarray(np.zeros((n, m, o), dtype=np.float32))
    if rank == 0:
        field_splitted = np.dsplit(field, size)
    else:
        field_splitted = None
    buff = comm.scatter(field_splitted, root=0)  # scatter the data
    local_field = np.zeros((n, m, o + 2), dtype=np.float32)  # need two ghost rows!
    local_field[:, :, 1:-1] = buff  # copy data to non-ghost rows
    local_field0 = np.zeros_like(local_field)  # array for previous time step
    local_field0[:] = local_field[:]
    return local_field0,local_field
