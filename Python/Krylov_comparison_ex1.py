import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys

from Krylov_solve import Krylov_solve
from solver.etdrdpif.wrapper import wrap_Krylov, wrap_solve


def main(solver, discretization=None):

    if solver == 'Krylov':
        discretization = None
        solve = Krylov_solve
    else:
        solve = wrap_Krylov
        if discretization is None:
            discretization = 'central'

    # Parameters of the specific experiment
    square_len = 2 * np.pi
    te = 1.0
    d = 1.0
    Diff = [d / 3, d / 3]
    a = 3.0
    Adv = [a/3, a/3]

    b = 100.0
    c = 1.0

    boundary = 'periodic'

    def u0(x, y, z):
        return 2 * np.cos(x + y + z), (b - c) * np.cos(x + y + z)

    def F(U1, V1):
        return -b * U1 + V1, -c * V1

    # runtime_Krylov, U_sol_Krylov = Krylov_solve(te, dt, steps, square_len, Adv, Diff, F, u0)

    runtime_rdp, U_sol_rdp = wrap_Krylov(te, 0.005, 10, square_len, Adv, Diff, F, u0)

    # print(np.max(U_sol_Krylov.flatten() - U_sol_rdp.flatten()))

    x = np.linspace(0, square_len, 10 + 1)[:-1]
    nodes = np.array(np.meshgrid([x], [x], [x], indexing='ij'))
    Uex = (np.exp(-b - d) + np.exp(-c - d)) * np.cos(np.sum(nodes, axis=0) - a).flatten()

    # print(np.max(U_sol_Krylov-Uex))
    print(np.max(U_sol_rdp-Uex))

    a_vals = [0, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 100, 200, 500, 1000]
    n = 5
    errors_euclid = np.zeros((len(a_vals), n))
    errors_max = np.zeros((len(a_vals), n))
    orders_euclid = np.zeros((len(a_vals), n - 1))
    orders_max = np.zeros((len(a_vals), n - 1))
    runtimes = np.zeros((len(a_vals), n))
    params = np.zeros((len(a_vals), n, 3))
    for i, a in enumerate(a_vals):
        err_old = np.inf
        err_max_old = np.inf
        Adv = [a/3.0, a/3.0]
        for j, h_exp in enumerate(range(n)):
            steps = 10*2**h_exp
            k = 0.005/(2**h_exp)
            x = np.linspace(0, square_len, steps + 1)[:-1]
            nodes = np.array(np.meshgrid([x], [x], [x], indexing='ij'))
            Uex = (np.exp(-b - d) + np.exp(-c - d)) * np.cos(np.sum(nodes, axis=0) - a).flatten()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    runtime, sol = solve(te, k, steps, square_len, Adv, Diff, F, u0, boundary,
                                         discretization=discretization)
                except Exception as e:
                    print("Error occurred")
                    print(e)
                    continue
            err = (Uex - sol).flatten()
            err_euclid = np.linalg.norm(err)
            err_max = np.linalg.norm(err, np.inf)
            order = np.log2(err_old / err_euclid)
            h = 2*np.pi/steps
            print(f"a={a}, C={a*k/h}, h={h}, k={k},\t error={err_max}, order={order}")
            errors_euclid[i, j] = err_euclid
            errors_max[i, j] = err_max
            runtimes[i, j] = runtime
            params[i, j, 0] = a
            params[i, j, 1] = h
            params[i, j, 2] = k
            if j > 0:
                orders_euclid[i, j-1] = order
                orders_max[i, j-1] = np.log2(err_max_old/err_max)
            err_old = err_euclid
            err_max_old = err_max
            sys.stdout.flush()
    sys.stdout.flush()
    experiment = f"Bhatt_ex1_{solver}_{discretization}"
    np.save(f"{experiment}_errors_euclid", errors_euclid)
    np.save(f"{experiment}_errors_max.npy", errors_max)
    np.save(f"{experiment}_orders_euclid.npy", orders_euclid)
    np.save(f"{experiment}_orders_max.npy", orders_max)
    np.save(f"{experiment}_runtimes.npy", runtimes)
    np.save(f"{experiment}_params_ahk.npy", params)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise ValueError("Pass at least the solver")
    args = sys.argv[1:]
    if len(args) < 2:
        args.append(None)
    main(*args)