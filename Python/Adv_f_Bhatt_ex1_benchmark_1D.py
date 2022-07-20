import sys

import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import profile

from etdrdpif.solve import etd_solve
from etdrdpif.discretize_periodic import discretize_periodic, discretize_upwind_periodic, discretize_upwind_Fromm_periodic,\
                                         discretize_upwind_thirdorder_periodic


# Benchmark problem 1D, 1 species
def benchmark_simple(dt, steps, a=1.0, out=False):
    # dt: time step. Default is 0.001
    # steps: number of spatial points in each coordinate direction. Default is 11

    # k is temporal discretization (dt); here: 0.005
    # h is spatial discretization (steps); here: 0.1

    dim = 1
    num_species = 2

    te = 1.0
    square_len = 2*np.pi


    ## Model Paramters and initial conditions
    #a = 3.0
    #d = 0.1
    d=1
    Adv = a*np.ones((num_species, dim))
    Diff = d*np.ones((num_species, dim))

    b = 100.0
    c = 1.0

    # Number of time steps
    tlen = int(np.floor(te/dt)) + 1
    # Discretize in space
    x, steps, nodes, A = discretize_periodic(steps, square_len, Diff, np.zeros_like(Adv))

    _, _, _, A_Adv = discretize_periodic(steps, square_len, np.zeros_like(Diff), Adv)

    #print(x, steps, nodes, A[0][0].todense())


    # Both species treated separately!
    # Possible due to assumption of no coupling in diffusive term
    # initial condition for u
    u_old = 2 * np.cos(np.sum(nodes, axis=1))
    v_old = (b - c) * np.cos(np.sum(nodes, axis=1))
    u_old = [u_old, v_old]

    u0 = u_old

    def F(u):
        f1 = -b*u[0] + u[1] + A_Adv[0][0] @ u[0]
        f2 = -c*u[1] + A_Adv[1][0] @ u[1]
        Fr = [f1, f2]
        return Fr

    runtime, soln = etd_solve(dt, tlen, steps, A, u0, F, save_all_steps=False)


    #u_soln = soln[-1, 0]

    u_soln = soln[0]
    v_soln = soln[1]

    print("Shape of nodes", nodes.shape)

    Uex = (np.exp((-b-d)*te)+np.exp((-c-d)*te))*np.cos(np.sum(nodes, axis=1)+a*te)
    Vex = (b-c)*np.exp((-c-d)*te)*np.cos(np.sum(nodes, axis=1)+a*te)


    # TODO: Sign of the advection term

    Uex = np.reshape(Uex, (steps, 1))
    u_ex = Uex
    Usoln = np.reshape(u_soln, (steps, 1))

    Vex = np.reshape(Vex, (steps, 1))
    v_ex = Vex
    Vsoln = np.reshape(v_soln, (steps, 1))

    print(np.max(np.abs(Usoln-Uex)))

    #print(Usoln.shape)
    #print(u_old)
    #print(Usoln)

    if out:

        # plt.imshow(soln[:, -1, :])
        # plt.show()

        plt.plot(nodes, Uex)
        plt.plot(nodes, Usoln)
        plt.show()

    return runtime, np.max(np.abs(Usoln-Uex)), np.linalg.norm((Usoln - Uex).flatten())/np.linalg.norm(u0[0].flatten())

    #if do_plot:
    #    plot_soln(Usoln, Vsoln, Uex, Vex, {x, x})


if __name__ == "__main__":
    #benchmark_simple(0.0005, 500, a=80, out=True)
    #exit()
    err_eucl_array = []
    err_max_array = []
    runtime_array = []
    orders_array = []
    parameters_array = []
    a_vals = [0.1, 0.5, 0.8, 1, 1.5, 2, 3, 5, 10, 20, 50, 100]
    for a in a_vals:
        print(f"a: {a}")
        h = 0.1
        k = 0.005
        errors_max = []
        errors_euclid = []
        runtimes = []
        parameters = []
        for i in range(0, 5):
            run, err_max, err_euclid = benchmark_simple(k, int(round(2*np.pi/h)), a, out=False)
            print(f"h={h}, k={k}")
            print("Errors:")
            print(f"max: {err_max}, euclid: {err_euclid}")
            errors_max.append(err_max)
            errors_euclid.append(err_euclid)
            runtimes.append(run)
            parameters.append((a, h, k))
            h /= 2
            k /= 2

        orders = []
        for i in range(len(errors_max)-1):
            orders.append(-np.log2(errors_max[i+1]/errors_max[i]))
        print("Orders")
        print(orders)

        err_eucl_array.append(errors_euclid)
        err_max_array.append(errors_max)
        orders_array.append(orders)
        runtime_array.append(runtimes)
        parameters_array.append(parameters)

    err_eucl_array = np.array(err_eucl_array)
    err_max_array = np.array(err_max_array)
    orders_array = np.array(orders_array)
    runtime_array = np.array(runtime_array)
    parameters_array = np.array(parameters_array)

    np.save("errors_euclid_bhatt1d", err_eucl_array)
    np.save("errors_max_bhatt1d", err_max_array)
    np.save("orders_bhatt1d", orders_array)
    np.save("runtimes_bhatt1d", runtime_array)
    np.save("parameters_ahk_bhatt1d", parameters_array)
