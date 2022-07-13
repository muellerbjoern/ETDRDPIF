import numpy as np
from etdrdpif.solve import etd_solve
from etdrdpif.discretize_periodic import discretize_periodic, discretize_upwind_periodic, \
    discretize_upwind_Fromm_periodic, \
    discretize_upwind_thirdorder_periodic


def wrap_solve(te, dt, steps, square_len, Adv, Diff, F, u0, boundary='periodic', discretization='central'):
    t = np.arange(0, te + dt, dt)
    tlen = len(t)

    if boundary == 'periodic':
        discretize_dict = {'central': discretize_periodic,
                           'fromm': discretize_upwind_Fromm_periodic,
                           'thirdorder': discretize_upwind_thirdorder_periodic}
        discretize = discretize_dict[discretization]
    if boundary == 'Neumann':
        raise NotImplementedError
    if boundary == 'Dirichlet':
        raise NotImplementedError
    if boundary == 'noflux':
        raise NotImplementedError

    # Discretize in space
    x, steps, nodes, A = discretize(steps, square_len, Diff, Adv)

    u_old = u0(nodes)

    runtime, soln = etd_solve(dt, tlen, steps, A, u_old, F, save_all_steps=False)

    return runtime, soln


def wrap_Krylov(te, dt, steps, square_len, Adv, Diff, F, u0, boundary='periodic', discretization='central'):

    Adv_full = np.ones((2, 3))
    Adv_full[0, :] = -Adv[0]
    Adv_full[1, :] = -Adv[1]

    Diff_full = np.ones((2, 3))
    Diff_full[0, :] = Diff[0]
    Diff_full[1, :] = Diff[1]

    def u0_new(nodes):
        return u0(nodes[:, 0], nodes[:, 1], nodes[:, 2])

    def F_new(u):
        return list(F(u[0], u[1]))

    runtime, soln = wrap_solve(te, dt, steps, square_len, Adv_full, Diff_full, F_new, u0_new, boundary, discretization)

    return runtime, soln[0]
