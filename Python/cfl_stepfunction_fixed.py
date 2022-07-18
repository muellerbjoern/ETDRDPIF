import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys

from etdrdpif.solve import etd_solve
from etdrdpif.discretize_periodic import discretize_periodic, discretize_upwind_periodic, \
    discretize_upwind_thirdorder_periodic, discretize_upwind_Fromm_periodic


# Benchmark problem 1D, 1 species
def benchmark_simple(dt, steps, a=1, out=False):
    # dt: time step. Default is 0.001
    # steps: number of spatial points in each coordinate direction. Default is 11

    # k is temporal discretization (dt); here: 0.005
    # h is spatial discretization (steps); here: 0.1

    dim = 1
    num_species = 1

    te = .1
    square_len = 1


    ## Model Paramters and initial conditions
    #a = -40.0
    #d = 0.1
    d=0
    Adv = a*np.ones((num_species, dim))
    Diff = d*np.ones((num_species, dim))

    # Discretize time interval
    tlen = int(np.floor(te/dt)) + 1

    # Discretize in space
    x, steps, nodes, A = discretize_periodic(steps, square_len, Diff, Adv)

    #print(x, steps, nodes, A[0][0].todense())

    def u0_func(x):
        a = x - np.floor(x)
        return np.logical_and(a >= .3, a <= .7)

    # Both species treated separately!
    # Possible due to assumption of no coupling in diffusive term
    # initial condition for u
    u0 = u0_func(nodes[:,0])
    u0 = [u0]

    def F(u):
        Fr = [np.zeros_like(u[0])]
        return Fr

    runtime, soln = etd_solve(dt, tlen, steps, A, u0, F, save_all_steps=False)
    u_soln = soln#[-1, 0]

    # TODO: Sign of the advection term
    Uex = u0_func(nodes[:,0]+a*te)

    Uex = np.reshape(Uex, (steps, 1))
    u_ex = Uex
    Usoln = np.reshape(u_soln, (steps, 1))

    #print(Usoln.shape)

    if out:

        #plt.imshow(soln[:, -1, :])
        #plt.show()

        plt.plot(nodes, Uex)
        plt.plot(nodes, Usoln)
        plt.savefig(f'cfl_stepfunction_C{a*dt*steps}_a{a}_h{1/steps}_dt{dt}_te{te}.eps')
        plt.show()
        plt.clf()

    return runtime, np.linalg.norm((Usoln-Uex).flatten())/ np.linalg.norm(u0[0]), np.max(np.abs((Usoln-Uex).flatten()))

    #if do_plot:
    #    plot_soln(Usoln, Vsoln, Uex, Vex, {x, x})


if __name__ == "__main__":

    # Here we see that after a certain point, reduction of k does not bring
    # any more improvement when we hold h fixed
    print("h fixed, reduce k")
    a = 100
    # ak/h = C => k = Ch/a
    h = 0.01/8
    err_old = np.inf
    for k_exp in range(12):
        k = 0.001*2**(-k_exp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, err_euclid, _ = benchmark_simple(k, int(round(1 / h)), a=a, out=True)
        order = np.log2(err_old / err_euclid)
        print(f"h={h}, k={k}, C={a * k / h},\t error={err_euclid}, order={order}")
        err_old = err_euclid
    sys.stdout.flush()
    # Here we test convergence for different values of the CFL number
    print("Letting h and k approach 0 for fixed CFL number")
    a = 100
    # ak/h = C => k = Ch/a
    a_vals = [0, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 100, 200, 500, 1000]
    n = 10
    errors_euclid = np.zeros((len(a_vals), n))
    errors_max = np.zeros((len(a_vals), n))
    orders_euclid = np.zeros((len(a_vals), n-1))
    runtimes = np.zeros((len(a_vals), n))
    params = np.zeros((len(a_vals), n, 3))
    for i, a in enumerate(a_vals[::-1]):
        err_old = np.inf
        for j, h_exp in enumerate(range(n)):
            h = 0.01*2**(-h_exp)
            k = h/20
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    runtime, err_euclid, err_max = benchmark_simple(k, int(round(1 / h)), a=a, out=True)
                except Exception as e:
                    print("Error occurred")
                    print(e)
            order = np.log2(err_old / err_euclid)
            print(f"a={a}, C={a*k/h}, h={h}, k={k},\t error={err_euclid}, order={order}")
            errors_euclid[i, j] = err_euclid
            errors_max[i, j] = err_max
            runtimes[i, j] = runtime
            params[i, j, 0] = a
            params[i, j, 1] = h
            params[i, j, 2] = k
            if j > 0:
                orders_euclid[i, j-1] = order
            err_old = err_euclid
            sys.stdout.flush()
    sys.stdout.flush()

    np.save("cfl_stepfunction_central_errors_euclid.npy", errors_euclid)
    np.save("cfl_stepfunction_central_errors_max.npy", errors_max)
    np.save("cfl_stepfunction_central_orders_euclid.npy", orders_euclid)
    np.save("cfl_stepfunction_central_runtimes.npy", runtimes)
    np.save("cfl_stepfunction_central_params_ahk.npy", params)

