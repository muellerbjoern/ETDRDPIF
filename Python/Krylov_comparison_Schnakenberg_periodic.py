import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys

import scipy.io

from Krylov_solve import Krylov_solve
from solver.etdrdpif.wrapper import wrap_Krylov, wrap_solve


def main(solver, discretization=None):

    if solver == 'Krylov':
        if discretization is None:
            discretization = 'expv'
        solve = Krylov_solve
    else:
        solve = wrap_Krylov
        if discretization is None:
            discretization = 'central'

    experiment = f"Bhatt_Schnakenberg_periodic_{solver}_{discretization}"

    # Parameters of the specific experiment
    square_len = 1.0
    te = 1.0
    Diff = [0.05, 0.01]
    A = 1.0

    b = 0.9

    gamma = 1.0

    Lambda = square_len

    boundary = 'periodic'

    def u0(x, y, z):
        u = 1 - np.exp(-10*((x-Lambda/2)**2 + (y-Lambda/2)**2 + (z-Lambda/2)**2))
        v = u + 0.1
        return u, v

    def F(U1, V1):
        return gamma*(A - U1 + U1**2*V1), gamma*(b - U1**2*V1)

    a_vals = [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 100, 200, 500, 1000]
    n = 6
    errors_euclid = np.zeros((len(a_vals), n))
    errors_max = np.zeros((len(a_vals), n))
    orders_euclid = np.zeros((len(a_vals), n - 1))
    orders_max = np.zeros((len(a_vals), n - 1))
    runtimes = np.zeros((len(a_vals), n+1))
    params = np.zeros((len(a_vals), n+1, 3))
    for i, a in enumerate(a_vals):
        Adv = [a, a]
        steps = 32
        k = 0.01
        try:
            runtime, sol = solve(te, k, steps, square_len, Adv, Diff, F, u0, boundary=boundary,
                                 discretization=discretization)
        except Exception as e:
            print("Error occurred")
            continue
        np.save(f"{experiment}_soln_a{a}_h0.01_over_{2**0}", sol)
        # print(np.min(sol), np.max(sol))
        # matsoln = scipy.io.loadmat("../MATLAB/matsoln.mat")["usoln"]
        # matsoln = matsoln.flatten()
        # diff = sol - matsoln
        # diff_max = np.max(np.abs(diff))
        runtimes[i, 0] = runtime
        params[i, 0] = np.array([a, 1.0/steps, k])
        err_old = np.inf
        err_max_old = np.inf
        for j, h_exp in enumerate(range(1, n)):
            #steps = 10*2**h_exp
            k = k/2
            #k = 0.005/(2**h_exp)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    print(k, steps)
                    runtime, sol_new = solve(te, k, steps, square_len, Adv, Diff, F, u0, boundary,
                                         discretization=discretization)
                    np.save(f"{experiment}_soln_a{a}_h0.01_over_{2 ** h_exp}", sol_new)
                    # runtime_etd, sol_etd = wrap_Krylov(te, k, steps, square_len, Adv, Diff, F, u0, boundary,
                    #                      discretization=discretization)
                    # print("Difference", np.max(np.abs(sol_etd-sol_new)))
                    # print(np.min(sol_new), np.max(sol_new))
                except Exception as e:
                    print("Error occurred")
                    print(e)
                    break
            err = (sol_new - sol).flatten()
            sol = sol_new
            err_euclid = np.linalg.norm(err)
            err_max = np.max(np.abs(err))
            order = np.log2(err_old / err_euclid)
            h = 1.0/steps
            print(f"a={a}, C={a*k/h}, h={h}, k={2*k},\t error={err_max}, order={order}")
            errors_euclid[i, j] = err_euclid
            errors_max[i, j] = err_max
            runtimes[i, j+1] = runtime
            params[i, j+1, 0] = a
            params[i, j+1, 1] = h
            params[i, j+1, 2] = k
            if j > 0:
                orders_euclid[i, j-1] = order
                orders_max[i, j-1] = np.log2(err_max_old/err_max)
            err_old = err_euclid
            err_max_old = err_max
            sys.stdout.flush()
    sys.stdout.flush()
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