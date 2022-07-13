import matplotlib.pyplot as plt
import numpy as np

from Krylov_solve import Krylov_solve
from solver.etdrdpif.wrapper import wrap_Krylov, wrap_solve


def main():
    square_len = 2 * np.pi
    te = 1.0
    d = 1.0
    Diff = [d / 3, d / 3]
    a = 1.0
    Adv = [a/3, a/3]

    steps = 30
    dt = 0.009
    b = 100.0
    c = 1.0

    def u0(x, y, z):
        return 2 * np.cos(x + y + z), (b - c) * np.cos(x + y + z)

    def F(U1, V1):
        return -b * U1 + V1, -c * V1

    runtime, U_sol = Krylov_solve(te, dt, steps, square_len, Adv, Diff, F, u0)

    # def u0(nodes):
    #     return [2*np.cos(np.sum(nodes, axis=1)), (b-c)*np.cos(np.sum(nodes, axis=1))]
    #
    # def F(u):
    #     return [-b*u[0] + u[1], - c*u[1]]
    #
    #
    # Adv_full = np.ones((2, 3))
    # Adv_full[0, :] = Adv[0]
    # Adv_full[1, :] = Adv[1]
    #
    # Diff_full = np.ones((2, 3))
    # Diff_full[0, :] = Diff[0]
    # Diff_full[1, :] = Diff[1]

    runtime_rdp, U_sol_rdp = wrap_Krylov(te, dt, steps, square_len, Adv, Diff, F, u0)

    print(U_sol.shape, U_sol_rdp.shape)
    print(np.max(U_sol.flatten() - U_sol_rdp.flatten()))

    x = np.linspace(0, square_len, steps + 1)[:-1]
    nodes = np.array(np.meshgrid([x],[x],[x], indexing='ij'))  # TODO: Check if this is really desired
    print(nodes.shape)
    Uex = (np.exp(-b-d)+np.exp(-c-d))*np.cos(np.sum(nodes, axis=0)-a).flatten()

    print(np.max(U_sol-Uex))
    print(np.max(U_sol_rdp-Uex))

if __name__ == '__main__':
    main()