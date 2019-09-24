import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('solve_matrix.pkl', 'rb') as f:
    solve= pickle.load(f)
solve_matrix=solve[0]

N=int(np.sqrt(len(solve_matrix.y[:,-1])))
omega_vector = solve_matrix.y[:,-1]
omega = np.reshape(omega_vector, ([N, N]))

plt.contourf(omega,levels=500)