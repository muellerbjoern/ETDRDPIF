import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import profile

from etdrdpif.solve import etd_solve
from etdrdpif.discretize_periodic import discretize_periodic, discretize_upwind_periodic

# Benchmark problem 1D, 1 species

@profile
def benchmark_simple(dt, steps, out=False):
    # dt: time step. Default is 0.001
    # steps: number of spatial points in each coordinate direction. Default is 11

    # k is temporal discretization (dt); here: 0.005
    # h is spatial discretization (steps); here: 0.1

    dim = 1
    num_species = 1

    te = 1
    square_len = 1


    ## Model Paramters and initial conditions
    a = -30.0
    #d = 0.1
    d=0
    Adv = a*np.ones((num_species, dim))
    Diff = d*np.ones((num_species, dim))

    # Discretize time interval
    t = np.arange(0, te+dt, dt)
    tlen = len(t)

    # Discretize in space
    x, steps, nodes, A = discretize_upwind_periodic(steps, square_len, Diff, Adv)

    #print(x, steps, nodes, A[0][0].todense())

    # Both species treated separately!
    # Possible due to assumption of no coupling in diffusive term
    # initial condition for u
    u_old = np.sin(2*np.pi*np.sum(nodes, axis=1))
    print(u_old)
    # initial condition for v
    u_old = [u_old]

    def F(u):
        Fr = [np.zeros_like(u[0])]
        return Fr

    runtime, soln = etd_solve(dt, tlen, steps, A, u_old, F, save_all_steps=True)


    u_soln = soln[-1, 0]
    # TODO: Sign of the advection term
    Uex = np.sin(2*np.pi*(np.sum(nodes, axis=1)+a*te))*np.exp(-d*4*np.pi*np.pi*te)

    Uex = np.reshape(Uex, (steps, 1))
    u_ex = Uex
    Usoln = np.reshape(u_soln, (steps, 1))

    print(Usoln.shape)

    if out:
        print(max(max(Usoln - Uex)))

        plt.imshow(soln[:, -1, :])
        plt.show()

        plt.plot(nodes, Uex)
        plt.plot(nodes, Usoln)
        plt.show()

    return max(max(Usoln-Uex))

    #if do_plot:
    #    plot_soln(Usoln, Vsoln, Uex, Vex, {x, x})


if __name__ == "__main__":
    benchmark_simple(0.0005, 1000, out=True)
    exit(0)
    sweep_k = []
    ks = 0.0002*np.logspace(-4, 4,num=20, base=2)
    ks = [0.0002/16]
    for k in ks:
        sweep_k.append(benchmark_simple(k, 1000))

    print(list(zip(ks, sweep_k)))
    # plt.plot(ks, sweep_k)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    print("a")
