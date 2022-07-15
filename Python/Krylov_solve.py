import numpy as np
import time

import scipy.io
import scipy.sparse as sp
from expv import expv


def Krylov_solve(te, dt, steps, square_len, Adv, Diff, F, u0, boundary='periodic', **kwargs):
    x0 = 0
    xn = square_len
    t0 = 0; tn = te; d1 = Diff[0] ;d2 = Diff[1]
    a1 = Adv[0]
    a2 = Adv[1]
    nn = steps
    k = dt # timestep
    if boundary == 'periodic':
        x = np.linspace(x0,xn,steps+1)
        x = x[:-1]
    if boundary == 'Neumann':
        x = np.linspace(x0, xn, steps+1)
    if boundary == 'Dirichlet':
        x = np.linspace(x0, xn, steps+1)
        x = x[1:-1]
    h = abs(x[0]-x[1])
    # print(h)

    y = x
    z = y
    # t = t0:k:tn # time discretization
    n = len(x)   #n points along x direction
    m = len(y)  #m points along y direction
    # K = len(t)  #k points along time direction
    #**************************construction of band matrix*****************
    e = np.ones((n,))
    B = sp.spdiags([ (2*d1+a1*h)*e, -4*d1*e, (2*d1-a1*h)*e],[-1, 0, 1],n,n, format='lil') # tridiagonal matrix
    if boundary == 'periodic':
        B[0,-1] = (2*d1+a1*h)
        B[-1,0] = (2*d1-a1*h)
    if boundary == 'Neumann':
        B[0, 1] = 4*d1
        B[-1, -2] = 4*d1
    if boundary == 'Dirichlet':
        pass  # Explicitly do nothing
    # D = np.ones(1, 1)*2*d1
    # A = -np.ones(1, 1)*2*a1
    # [x2, steps2, nodes2, B2] = discretize_periodic(steps, xn, D, A);
    # B2 = -B2{1, 1}*h^2;
    # disp(B2);
    # disp(size(B));
    # disp(["B2", size(B2)]);
    # disp(B+B2);
    # disp(steps); disp(steps2); disp(x); disp(x2);
    B1 = sp.spdiags([ (2*d2+a2*h)*e, -4*d2*e, (2*d2-a2*h)*e],[-1, 0, 1],n,n, format='lil') #tridiagonal matrix
    if boundary == 'periodic':
        B1[0,-1] = (2*d2+a2*h)
        B1[-1,0] = (2*d2-a2*h)
    if boundary == 'Neumann':
        B1[0, 1] = 4*d2
        B1[-1, -2] = 4*d2
    if boundary == 'Dirichlet':
        pass  # Explicitly do nothing

    # B_mat = scipy.io.loadmat('../MATLAB/B.mat')['B']
    # B1_mat = scipy.io.loadmat('../MATLAB/B1.mat')['B1']
    #
    # print(np.max(np.abs(B-B_mat)))
    # print(np.max(np.abs(B1-B1_mat)))

    A1 = 1/(2*h**2)*( sp.kron(B, sp.kron(sp.eye(n), sp.eye(n))) + sp.kron(sp.eye(n),
        sp.kron(B, sp.eye(n)))+ sp.kron(sp.eye(n), sp.kron(sp.eye(n),B)))
    A2 = 1/(2*h**2)*( sp.kron(B1, sp.kron(sp.eye(n), sp.eye(n))) + sp.kron(sp.eye(n),
        sp.kron(B1, sp.eye(n)))+ sp.kron(sp.eye(n), sp.kron(sp.eye(n),B1)))

    # %********************************************************************
    U = np.zeros((n,n,n))
    V = np.zeros((n,n,n))
    # initial condition set up
    for p in range(n):
        for q in range(n):
            for i in range(n):
                U[p, q, i], V[p, q, i] = u0(x[p], y[q], z[i])#2.0*np.cos(x[p] + y[q] + z[i])
                # V[p, q, i] = (b-c)*np.cos(x[p] + y[q] + z[i])
    U_1 = U.flatten(); V_1 = V.flatten()
    U_2 = U_1; V_2 = V_1
    U_3 = U_1; V_3 = V_1
    M1 = int(np.floor((tn-t0)/(2*k))) + 1 # TODO: If results differ from MATLAB, check M1 again
    # T = t0:2*k:tn; M1 = len(T)
    m1 = 10 # Krylov subspace dimension
    start = time.time()

    for l in range(1, M1):
        # mat = scipy.io.loadmat(f"../MATLAB/U1{l + 1}.mat")['U_1']
        # mat = mat.flatten()
        # diff = U_1 - mat
        # U_1 = mat
        # print(np.max(np.abs(diff)))
        # mat = scipy.io.loadmat(f"../MATLAB/V1{l + 1}.mat")['V_1']
        # mat = mat.flatten()
        # diff = V_1 - mat
        # V_1 = mat
        # print('V', np.max(np.abs(diff)))
        F1, G1 = F(U_1, V_1)
        # mat = scipy.io.loadmat(f"../MATLAB/F1{l + 1}.mat")['F1']
        # mat = mat.flatten()
        # diff = F1 - mat
        # print('F', np.max(np.abs(diff)))
        # mat = scipy.io.loadmat(f"../MATLAB/G1{l + 1}.mat")['G1']
        # mat = mat.flatten()
        # diff = G1 - mat
        # print('G', np.max(np.abs(diff)))
        U_1 = expv( k, A1, ( U_1 + k*F1), 1.0e-7, krylov_dim=m1)
        V_1 = expv( k, A2, ( V_1 + k*G1),1.0e-7,krylov_dim=m1)

        # mat = scipy.io.loadmat(f"../MATLAB/U1_2{l + 1}.mat")['U_1']
        # mat = mat.flatten()
        # diff = U_1 - mat
        # print('U1_2', np.max(np.abs(diff)))
        # mat = scipy.io.loadmat(f"../MATLAB/V1_2{l + 1}.mat")['V_1']
        # mat = mat.flatten()
        # diff = V_1 - mat
        # print('V1_2', np.max(np.abs(diff)))
        F1, G1 = F(U_1, V_1)
        U_1 = expv( k, A1, (U_1 + k*F1),1.0e-7,krylov_dim=m1)
        V_1 = expv( k, A2, ( V_1 + k*G1),1.0e-7,krylov_dim=m1)

        # mat = scipy.io.loadmat(f"../MATLAB/U1_3{l + 1}.mat")['U_1']
        # mat = mat.flatten()
        # diff = U_1 - mat
        # U_1 = mat
        # print('U1_3', np.max(np.abs(diff)))
        # mat = scipy.io.loadmat(f"../MATLAB/V1_3{l + 1}.mat")['V_1']
        # mat = mat.flatten()
        # diff = V_1 - mat
        # V_1 = mat
        # print('V1_3', np.max(np.abs(diff)))
        # Extrapolation Scheme

        F2, G2 = F(U_2, V_2)

        U_2 = expv( 2*k, A1, (U_2 + 2*k*F2),1.0e-7,krylov_dim=m1)
        V_2 = expv( 2*k, A2, (V_2 + 2*k*G2),1.0e-7,krylov_dim=m1)

        # mat = scipy.io.loadmat(f"../MATLAB/U2{l + 1}.mat")['U_2']
        # mat = mat.flatten()
        # diff = U_2 - mat
        # U_2 = mat
        # print('U_2', np.max(np.abs(diff)))
        # mat = scipy.io.loadmat(f"../MATLAB/V2{l + 1}.mat")['V_2']
        # mat = mat.flatten()
        # diff = V_2 - mat
        # V_2 = mat
        # print('V_2', np.max(np.abs(diff)))
        F3, G3 = F(U_3, V_3)

        U_3 = expv( 2*k, A1, (U_3 + 2*k*F3),1.0e-7,krylov_dim=m1)
        V_3 = expv( 2*k, A2, (V_3 + 2*k*G3),1.0e-7,krylov_dim=m1)
        # mat = scipy.io.loadmat(f"../MATLAB/U3{l + 1}.mat")['U_3']
        # mat = mat.flatten()
        # diff = U_3 - mat
        # U_3 = mat
        # print('U_3', np.max(np.abs(diff)))
        # mat = scipy.io.loadmat(f"../MATLAB/V3{l + 1}.mat")['V_3']
        # mat = mat.flatten()
        # diff = V_3 - mat
        # V_3 = mat
        # print('V_3', np.max(np.abs(diff)))

        sol1 = 2*U_1-(U_2+U_3)/2
        sol2 = 2*V_1-(V_2+V_3)/2
        # Extrapolation scheme

        # mat = scipy.io.loadmat(f"../MATLAB/sol1{l + 1}.mat")['sol1']
        # mat = mat.flatten()
        # diff = sol1 - mat
        # sol1 = mat
        # print('sol1', np.max(np.abs(diff)))
        # mat = scipy.io.loadmat(f"../MATLAB/sol2{l + 1}.mat")['sol2']
        # mat = mat.flatten()
        # diff = sol2 - mat
        # sol2 = mat
        # print('sol2', np.max(np.abs(diff)))
        U_1 = sol1;V_1 = sol2;U_2 = U_1
        V_2 = V_1;U_3 = U_1; V_3 = V_1

    runtime = time.time()-start
    u_soln = sol1

    return runtime, u_soln
