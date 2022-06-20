import matplotlib.pyplot as plt
import numpy as np

from etdrdpif.solve import etd_solve
from etdrdpif.discretize_periodic import discretize_periodic, discretize_upwind_periodic


# Benchmark problem 1D, 1 species
def benchmark_simple(dt, steps, out=False):
    # dt: time step. Default is 0.001
    # steps: number of spatial points in each coordinate direction. Default is 11

    # k is temporal discretization (dt); here: 0.005
    # h is spatial discretization (steps); here: 0.1

    dim = 1
    num_species = 1

    te = .1
    square_len = 1


    ## Model Paramters and initial conditions
    a = -2.0
    #d = 0.1
    d=0
    Adv = a*np.ones((num_species, dim))
    Diff = d*np.ones((num_species, dim))

    # Discretize time interval
    t = np.arange(0, te+dt, dt)
    tlen = len(t)

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
    Uex = np.sin(2*np.pi*(np.sum(nodes, axis=1)+a*te))*np.exp(-d*4*np.pi*np.pi*te)
    Uex = u0_func(nodes+a*te)*np.exp(-d*4*np.pi*np.pi*te)

    Uex = np.reshape(Uex, (steps, 1))
    u_ex = Uex
    Usoln = np.reshape(u_soln, (steps, 1))

    #print(Usoln.shape)

    if out:

        #plt.imshow(soln[:, -1, :])
        #plt.show()

        plt.plot(nodes, Uex)
        plt.plot(nodes, Usoln)
        plt.savefig(f'{dt}_{1/steps}_{a}_{d}.eps')
        plt.show()

    return np.linalg.norm((Usoln-Uex).flatten())/ np.linalg.norm(u0[0])

    #if do_plot:
    #    plot_soln(Usoln, Vsoln, Uex, Vex, {x, x})


if __name__ == "__main__":
    ks = 0.0002*np.logspace(-7, 4,num=12, base=2)
    hs_C_max = [2*k for k in ks]
    hs_C_low = [4*k for k in ks]
    hs_C_high = [1*k for k in ks]
    sweep_k = []
    for k, h in zip(ks, hs_C_low):
        print(k, int(round(1/h)))
        sweep_k.append(benchmark_simple(k, int(round(1/h)), out=True))

    print(list(zip(ks, sweep_k)))
    plt.plot(ks, sweep_k)
    plt.xscale('log', base=2)
    plt.yscale('log', base=2)
    plt.savefig("errors_C_low.png")
    #plt.show()

    orders = []
    for i in range(len(sweep_k)-1):
        orders.append(np.log2(sweep_k[i+1]/sweep_k[i])/np.log2(ks[i+1]/ks[i]))
    print(orders)
