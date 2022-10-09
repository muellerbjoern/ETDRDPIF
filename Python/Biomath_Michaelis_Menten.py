import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys

import scipy.io


from Krylov_solve import Krylov_solve
from solver.etdrdpif.wrapper import wrap_Krylov, wrap_solve
from solver.etdrdpif.solve import etd_solve

from solver.etdrdpif.discretize_periodic import discretize_upwind_Fromm_Dirichlet_mirror, discretize_upwind_Fromm_Dirichlet_centralboundary, discretize_upwind_Fromm_Dirichlet_outsidezero, discretize_upwind_Fromm_Dirichlet_derivativezero

square_len = 1.0
te = 1.0
Diff = np.empty((1,1))
Diff[0, 0] = 0.01
Adv = np.empty((1,1))
Adv[0, 0] = .1

def u0(x):
    return [np.ones_like(x)]

def F(u):
    return [-u[0]/(1+u[0])]

steps = 100
dt = 0.001

tlen = int(np.floor(te / dt)) + 1
x, steps, nodes, A = discretize_upwind_Fromm_Dirichlet_mirror(steps, square_len, Diff, Adv)

u_old = u0(x)

runtime, soln = etd_solve(dt, tlen, steps, A, u_old, F, save_all_steps=True)

# print(soln)
# print(soln.shape)
# plt.plot(soln[-1, 0, :])
# plt.show()

soln2 = np.zeros((*soln.shape[:-1], soln.shape[-1]+2))
print(soln2.shape)
soln2[:, :, 1:-1] = soln
soln = soln2

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

Y, X = np.meshgrid(np.arange(tlen), np.arange(steps+2))
from matplotlib import cm
surf = ax.plot_surface(X, Y, soln[:, 0, :].T, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.show()