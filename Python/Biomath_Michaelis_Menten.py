import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys

import scipy.io


from Krylov_solve import Krylov_solve
from solver.etdrdpif.wrapper import wrap_Krylov, wrap_solve
from solver.etdrdpif.solve_nosplit import etd_solve

from solver.etdrdpif.discretize_periodic import discretize_upwind_Fromm_Dirichlet_mirror, discretize_upwind_Fromm_Dirichlet_centralboundary, discretize_upwind_Fromm_Dirichlet_outsidezero, discretize_upwind_Fromm_Dirichlet_derivativezero

square_len = 1.0
te = 1.0
Diff = np.empty((1,3))
Diff[0, :] = 0.01
Adv = np.empty((1,3))
Adv[0, :] = .1

def u0(nodes):
    x = nodes[:, 0]
    return [np.ones_like(x)]

def F(u):
    return [-u[0]/(1+u[0])]

steps = 100
dt = 0.001

#tlen = int(np.floor(te / dt)) + 1
#x, steps, nodes, A = discretize_upwind_Fromm_Dirichlet_mirror(steps, square_len, Diff, Adv)

#u_old = u0(x, x, x)

#runtime, soln = etd_solve(dt, tlen, steps, A, u_old, F, save_all_steps=True)

a_vals = [0, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 100, 200, 500, 1000]
n = 4
errors_euclid = np.zeros((len(a_vals), n))
errors_max = np.zeros((len(a_vals), n))
orders_euclid = np.zeros((len(a_vals), n - 1))
orders_max = np.zeros((len(a_vals), n - 1))
runtimes = np.zeros((len(a_vals), n+1))
params = np.zeros((len(a_vals), n+1, 3))
for i, a in enumerate(a_vals):
    Adv[0, :] = a
    steps = 32
    k = 0.01
    start = 0
    while True:
        try:
            tlen = int(np.floor(te / k)) + 1
            x, steps, nodes, A = discretize_upwind_Fromm_Dirichlet_mirror(steps, square_len, Diff, Adv)

            u_old = u0(nodes)
            print("u", u_old[0].shape)

            runtime, sol = etd_solve(k, tlen, steps, A, u_old, F, save_all_steps=True)
        except Exception as e:
            raise e
            print("Error occurred", e)
            k = k/2
            start += 1
        else:
            break
    # np.save(f"{experiment}_soln_a{a}_h0.01_over_{2**start}", sol)
    print(a, np.min(sol), np.max(sol), runtime)
    # matsoln = scipy.io.loadmat("../MATLAB/matsoln.mat")["usoln"]
    # matsoln = matsoln.flatten()
    # diff = sol - matsoln
    # diff_max = np.max(np.abs(diff))
    runtimes[i, start] = runtime
    params[i, start] = np.array([a, 1.0/steps, k])
    err_old = np.inf
    err_max_old = np.inf
    for j, h_exp in enumerate(range(start+1, n)):
        #steps = 10*2**h_exp
        k = k/2
        #k = 0.005/(2**h_exp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                print(k, steps)
                tlen = int(np.floor(te / k)) + 1
                x, steps, nodes, A = discretize_upwind_Fromm_Dirichlet_mirror(steps, square_len, Diff, Adv)

                u_old = u0(nodes)
                print(u_old[0].shape)

                runtime, sol_new = etd_solve(k, tlen, steps, A, u_old, F, save_all_steps=False)
                np.save(f"michment_soln_a{a}_h0.01_over_{2 ** h_exp}", sol_new)

                print(sol_new[0].shape)

                sol_plot = sol_new[0].reshape((steps, steps, steps))
                x = np.linspace(0, square_len, steps+2)[1:-1]
                y = x
                nodes_x, nodes_y = np.meshgrid(x, y)
                fig = plt.figure()
                ax = plt.axes(projection='3d')
                ax.plot_surface(nodes_x, nodes_y, sol_plot[:, :, 0])
                plt.show()
                # runtime_etd, sol_etd = wrap_Krylov(te, k, steps, square_len, Adv, Diff, F, u0, boundary,
                #                      discretization=discretization)
                # print("Difference", np.max(np.abs(sol_etd-sol_new)))
                print(np.min(sol_new), np.max(sol_new))
            except Exception as e:
                raise e
                print("Error occurred")
                print(e)
                break
        err = (sol_new[0] - sol[0]).flatten()
        sol = sol_new
        err_euclid = np.linalg.norm(err)
        err_max = np.max(np.abs(err))
        order = np.log2(err_old / err_euclid)
        h = 1.0/steps
        print(f"a={a}, C={a*k/h}, h={h}, k={2*k},\t error={err_max}, order={order}, order_max={np.log2(err_max_old/err_max)}")
        errors_euclid[i, start+j] = err_euclid
        errors_max[i, start+j] = err_max
        runtimes[i, start+j+1] = runtime
        params[i, start+j+1, 0] = a
        params[i, start+j+1, 1] = h
        params[i, start+j+1, 2] = k
        if j > 0:
            orders_euclid[i, start+j-1] = order
            orders_max[i, start+j-1] = np.log2(err_max_old/err_max)
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