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
    ks = 0.0002*np.logspace(-5, 4,num=12, base=2)
    hs_C_max = [2*k for k in ks]
    hs_C_low = [4*k for k in ks]
    hs_C_high = [1*k for k in ks]
    sweep_k = []
    for k, h in zip(ks, hs_C_high):
        print(k, int(round(1/h)))
        sweep_k.append(benchmark_simple(k, int(round(1/h)), out=True))

    print(list(zip(ks, sweep_k)))
    plt.plot(ks, sweep_k)
    plt.xscale('log', basex=2)
    plt.yscale('log', basey=2)
    plt.savefig("errors_C_low.png")
    plt.show()

    # One result: [(6.25e-06, 0.02099183913608325), (1.1019890687450267e-05, 0.024743537856384692), (1.9430078522136498e-05, 0.029893367999905055), (3.42587746180031e-05, 0.03668396917696528), (6.040447222022238e-05, 0.042609682667666966), (0.00010650410894399628, 0.052708002161481506), (0.00018778618213234128, 0.061032608218402666), (0.0003311013119539244, 0.07060974966179123), (0.0005837920422725786, 0.0855021507678495), (0.001029331918407546, 0.10053618667904084), (0.00181490003551274, 0.12331839175002765), (0.0032, 0.1481004373642344)]

