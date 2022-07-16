"""
A Python implementation of expm from Expokit
"""

import numpy as np
from numpy.linalg import norm

from scipy.sparse.linalg import norm as spnorm
from scipy.linalg import expm
import scipy.sparse.linalg


def expv(t, A, v, tol=1e-7, krylov_dim=30):
    n, _ = A.shape
    m = min(krylov_dim, n)

    anorm = scipy.sparse.linalg.norm(A, np.inf);
    mxrej = 10;  btol  = 1.0e-7;
    gamma = 0.9; delta = 1.2;
    mb    = m; t_out   = abs(t);
    nstep = 0; t_new   = 0;
    t_now = 0; s_error = 0;

    k1 = 2; xm = 1/m; normv = np.linalg.norm(v); beta = normv;
    fact = (((m+1)/np.exp(1))**(m+1))*np.sqrt(2*np.pi*(m+1));
    t_new = (1/anorm)*((fact*tol)/(4*beta*anorm))**xm;
    s = 10**(np.floor(np.log10(t_new))-1); t_new = np.ceil(t_new/s)*s;
    sgn = np.sign(t); nstep = 0;


    w = v;
    hump = normv;
    while t_now < t_out:
      nstep = nstep + 1;
      t_step = t_out; #min( t_out-t_now,t_new );
      V = np.zeros((n,m+1));
      H = np.zeros((m+2,m+2));

      V[:, 0] = (1/beta)*w
      for j in range(m):
         p = A@V[:,j]
         for i in range(j):
            H[i,j] = V[:,i].T@p;
            p = p-H[i,j]*V[:,i];
         s = np.linalg.norm(p);
         H[j+1,j] = s
         V[:,j+1] = (1/s)*p;
      if k1 != 0:
         H[m+1,m] = 1;
         avnorm = np.linalg.norm(A@V[:,m]);

      mx = m;
      F = expm(sgn*t_step*H[:mx,:mx]);

      mx = m; #mb + max( 0,k1-1 );
      w = V[:,:mx]@(beta*F[:mx,0]);
      beta = np.linalg.norm( w );
      hump = max(hump,beta);

      t_now = t_now + t_step;

    return w


def old_expv(t, A, v, tol=1e-7, krylov_dim=30):
    """
        Evaluates exp(t * A) @ v efficiently using Krylov subspace projection
        techniques and matrix-vector product operations.
    This function implements the expv function of the Expokit library
    (https://www.maths.uq.edu.au/expokit). It is in particular suitable for
    large sparse matrices.
    Args:
    t (float): real or complex time-like parameter
    A (array or sparse): an numpy.array or scipy.sparse square matrix
    v (array): a vector with size compatible with A
    tol (real): the required tolerance (default 1e-7)
    krylov_dim (int): dimension of the Krylov subspace
                      (typically a number between 15 and 50, default 30)
    Returns:
    The array w(t) = exp(t * A) @ v.
    """
    assert A.shape[1] == A.shape[0], "A must be square"
    assert A.shape[1] == v.shape[0], "v and A must have compatible shapes"

    n = A.shape[0]
    m = min(krylov_dim, n)

    anorm = spnorm(A, ord=np.inf)

    out_type = np.result_type(type(t), A.dtype, v.dtype)

    # safety factors
    gamma = 0.9
    delta = 1.2

    btol = 1e-7     # tolerance for "happy-breakdown"
    maxiter = 10    # max number of time-step refinements

    rndoff = anorm*np.spacing(1)

    # estimate first time-step and round to two significant digits
    beta = norm(v)
    r = 1/m
    fact = (((m+1)/np.exp(1.0)) ** (m+1))*np.sqrt(2.0*np.pi*(m+1))
    tau = (1.0/anorm) * ((fact*tol)/(4.0*beta*anorm)) ** r

    outvec = np.zeros(v.shape, dtype=out_type)

    # storage for Krylov subspace vectors
    v_m = np.zeros((m + 1, len(v)), dtype=out_type)
    # for i in range(1, m + 2):
    #     vm.append(np.empty_like(outvec))
    h_m = np.zeros((m+2, m+2), dtype=outvec.dtype)

    t_f = np.abs(t)

    # For some reason numpy.sign has a different definition than Julia or MATLAB
    if isinstance(t, complex):
        tsgn = t / np.abs(t)
    else:
        tsgn = np.sign(t)

    t_k = 0. * t_f
    w = np.array(v, dtype=out_type)
    p = np.empty_like(w)

    m_x = m
    while t_k < t_f:
        tau = min(t_f - t_k, tau)
        tau = t_f - t_k
        # Arnoldi procedure
        v_m[0] = w / beta
        m_x = m

        for j in range(m):
            p = A.dot(v_m[j])

            h_m[:j+1, j] = v_m[:j+1, :].conj() @ p
            tmp = h_m[:j+1, j][:, np.newaxis] * v_m[:j+1]
            p -= np.sum(tmp, axis=0)

            s = norm(p)
            # if s < btol: # happy-breakdown
            #     tau = t_f - t_k
            #     err_loc = btol
            #
            #     f_m = expm(tsgn * tau * h_m[:j+1, :j+1])
            #
            #     tmp = beta * f_m[:j+1, 0][:, np.newaxis] * v_m[:j+1, :]
            #     w = np.sum(tmp, axis=0)
            #
            #     m_x = j
            #     break

            h_m[j+1, j] = s
            v_m[j+1] = p / s

        h_m[m + 1, m] = 1.

        if m_x == m:
            avnorm = norm(A @ v_m[m])

        # propagate using adaptive step size
        #i = 1
        #while i < maxiter and m_x == m:
        f_m = expm(tsgn * tau * h_m)

            # err1 = abs(beta * f_m[m, 0])
            # err2 = abs(beta * f_m[m+1, 0] * avnorm)
            #
            # if err1 > 10*err2:	# err1 >> err2
            #     err_loc = err2
            #     r = 1/m
            # elif err1 > err2:
            #     err_loc = (err1*err2)/(err1-err2)
            #     r = 1/m
            # else:
            #     err_loc = err1
            #     r = 1/(m-1)

            # is time step sufficient?
            # if err_loc <= delta * tau * (tau*tol/err_loc) ** r:
        tmp = beta * f_m[:m+1, 0][:, np.newaxis] * v_m[:m+1, :]
        w = np.sum(tmp, axis=0)
            #     break
            #
            # # estimate new time-step
            # tau = gamma * tau * (tau * tol / err_loc) ** r
            # i += 1

        # if i == maxiter:
        #     raise(RuntimeError("Number of iteration exceeded maxiter. "
        #                        "Requested tolerance might be too high."))

        beta = norm(w)
        t_k += tau
        #tau = gamma * tau * (tau * tol / err_loc) ** r # estimate new time-step
        #err_loc = max(err_loc, rndoff)

        h_m.fill(0.)

    return w