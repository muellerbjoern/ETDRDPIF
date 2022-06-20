import numpy as np
import scipy.sparse as sp

from etdrdpif.solve import etd_solve
from etdrdpif.discretize_periodic import discretize_periodic


# Benchmark problem from Bhatt et al., 2018, reduced to single dimension,
# but still 2 species!

def Bhatt_benchmark_1D(dt, steps):
    # dt: time step. Default is 0.001
    # steps: number of spatial points in each coordinate direction. Default is 11

    # k is temporal discretization (dt); here: 0.005
    # h is spatial discretization (steps); here: 0.1

    dim = 1
    num_species = 2

    te = 1.0
    square_len = 2*np.pi


    ## Model Paramters and initial conditions
    a = 3.0
    #d = 0.1
    d=1.0
    Adv = a/1.0*np.ones((num_species, dim))
    Diff = d/1.0*np.ones((num_species, dim))

    #b = 0.1;
    b = 100.0
    c = 1.0

    # Discretize time interval
    t = np.arange(0, te+dt, dt)
    tlen = len(t)

    # Discretize in space
    x, steps, nodes, A = discretize_periodic(steps, square_len, Diff, Adv)

    # Both species treated separately!
    # Possible due to assumption of no coupling in diffusive term
    # initial condition for u
    u_old = 2*np.cos(np.sum(nodes, axis=1))
    # initial condition for v
    v_old = (b-c)*np.cos(np.sum(nodes, axis=1))
    u_old = [u_old, v_old]

    def F(u):
        f0 = -b * u[0] + u[1]
        f1 = -c * u[1]
        Fr = [f0, f1]
        return Fr

    runtime, soln = etd_solve(dt, tlen, steps, A, u_old, F)

    u_soln = soln[0]
    v_soln = soln[1]

    Uex = (np.exp(-b-d)+np.exp(-c-d))*np.cos(np.sum(nodes, axis=1)-a)
    Vex = (b-c)*np.exp(-c-d)*np.cos(np.sum(nodes, axis=1)-a)

    Uex = np.reshape(Uex, (steps, 1))
    u_ex = Uex
    Vex = np.reshape(Vex, (steps, 1))
    Usoln = np.reshape(u_soln, (steps, 1))
    Vsoln = np.reshape(v_soln, (steps, 1))

    print(max(max(Usoln - Uex)))

    #if do_plot:
    #    plot_soln(Usoln, Vsoln, Uex, Vex, {x, x})


if __name__ == "__main__":
    Bhatt_benchmark_1D(0.00001, 100000)
