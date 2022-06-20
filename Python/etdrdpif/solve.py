
import time

import numpy as np
import scipy.sparse as sp
from memory_profiler import profile
from scipy.sparse import  linalg as splinalg

@profile
def etd_solve(dt, tlen, steps, A, u_old, F, save_all_steps=False):

    dim, num_species = len(A), len(A[0])

    if save_all_steps:
        all_steps = np.zeros((tlen, num_species, *u_old[0].shape))
        all_steps.fill(0)
        all_steps[0] = np.array(u_old)

    # System matrices
    r1 = 1/3
    r2 = 1/4
    # Identity matrix of correct size
    Id_temp = sp.eye(1, format='csc')
    I = sp.eye(steps, format='csc')
    for i_dim in range(dim):
        Id_temp = sp.kron(Id_temp, I)

    # Stack identity matrices analogously to A

    Id = []
    for i_dim in range(dim):
        Id.append([])
        for i_spec in range(num_species):
            Id[i_dim].append(Id_temp)

    A1 = []
    A2 = []
    A3 = []
    for i_dim in range(dim):
        A1.append([])
        A2.append([])
        A3.append([])
        for i_spec in range(num_species):
            print(Id[i_dim][i_spec].shape)
            print(A[i_dim][i_spec].shape)
            A1[i_dim].append(Id[i_dim][i_spec] + r1*dt*A[i_dim][i_spec])
            A2[i_dim].append(Id[i_dim][i_spec] + r2*dt*A[i_dim][i_spec])
            A3[i_dim].append(Id[i_dim][i_spec] + dt*A[i_dim][i_spec])

    # 3D equivalent
    # A1x = (Id_temp + r1*dt*Ax);
    # A2x = (Id_temp + r2*dt*Ax);
    # A3x = (Id_temp + dt*Ax);
    # A1y = (Id_temp + r1*dt*Ay);
    # A2y = (Id_temp + r2*dt*Ay);
    # A3y = (Id_temp + dt*Ay);
    # A1z = (Id_temp + r1*dt*Az);
    # A2z = (Id_temp + r2*dt*Az);
    # A3z = (Id_temp + dt*Az);

    del A, Id_temp, Id, I

    LU1 = []
    LU2 = []
    LU3 = []

    for i_dim in range(dim):
        LU1.append([])
        LU2.append([])
        LU3.append([])
        for i_spec in range(num_species):
            LU1[i_dim].append(splinalg.splu(A1[i_dim][i_spec]))
            LU2[i_dim].append(splinalg.splu(A2[i_dim][i_spec]))
            LU3[i_dim].append(splinalg.splu(A3[i_dim][i_spec]))

    del A1, A2, A3

    start_time = time.time()

    print("beginning iteration")

    for i in range(1, tlen):

        F_old = F(u_old)

        p = []
        d = []
        for i_spec in range(num_species):
            p.append(F_old[i_spec])  # TODO: Check dimensions
            d.append(u_old[i_spec])
            for i_dim in range(dim):
                # TODO: Aggregate RHS, might be faster due to BLAS routine?
                #p{i_spec} = U3{i_spec, i_dim}\(L3{i_spec, i_dim}\p{i_spec});
                #d{i_spec} = U3{i_spec, i_dim}\(L3{i_spec, i_dim}\d{i_spec});
                p_d = LU3[i_dim][i_spec].solve(np.array([p[i_spec], d[i_spec]]).T)
                p[i_spec] = p_d[:, 0]
                d[i_spec] = p_d[:, 1]

        u_star = [d[i_spec] + p[i_spec]*dt for i_spec in range(num_species)]
        F_star = F(u_star)
        # %     % For u
        # %     p1 = U3x\(L3x\F_old(:,1));
        # %     p2 = U3y\(L3y\p1);
        # %     p3u = U3z\(L3z\p2);
        # %     % For v
        # %     p1 = U3x\(L3x\F_old(:,2));
        # %     p2 = U3y\(L3y\p1);
        # %     p3v = U3z\(L3z\p2);
        # %
        # %     % For u
        # %     d1 = U3x\(L3x\u_old);
        # %     d2 = U3y\(L3y\d1);
        # %     d3u = U3z\(L3z\d2);
        # %     u_star = d3u + dt*p3u;
        # %     % For v
        # %     d1 = U3x\(L3x\v_old);
        # %     d2 = U3y\(L3y\d1);
        # %     d3v = U3z\(L3z\d2);
        # %     v_star = d3v + dt*p3v;
        # %     F_star = F(u_star,v_star);

        # Cell arrays to store intermediate results

        # Contains intermediate RHS needed to compute c4 (F_old, c2, c4)
        c4 = []
        # Contains intermediate RHS needed to compute c3 (u_old, c1, c3)
        c3 = []

        s1 = []
        s2 = []

        for i_spec in range(num_species):
            # Initialize RHS
            c4.append(F_old[i_spec])
            c3.append(u_old[i_spec])
            for i_dim in range(dim-1):
                # Solve for c4, linear system with F_old as RHS
                a1_b1 = LU1[i_dim][i_spec].solve(np.array([c3[i_spec][:, np.newaxis], c4[i_spec][:, np.newaxis]]))
                a1 = a1_b1[:, 0]
                b1 = a1_b1[:, 1]
                a2_b2 = LU2[i_dim][i_spec].solve(np.array([c3[i_spec][:, np.newaxis], c4[i_spec][:, np.newaxis]]))
                a2 = a2_b2[:, 0]
                b2 = a2_b2[:, 1]
                c4[i_spec] = 9*b1-8*b2

                # Solve for c3, linear system with u_old as RHS
                #a1 = U1{i_spec, i_dim}\(L1{i_spec, i_dim}\c3{i_spec});
                #a2 = U2{i_spec, i_dim}\(L2{i_spec, i_dim}\c3{i_spec});
                c3[i_spec]= 9*a1-8*a2

            # Summarize c3 and c4 to the summands of equation (19)
            # (Asante-Asamani, 2020)
            s1_arg = 9*c3[i_spec]+2*dt*c4[i_spec]+dt*F_star[i_spec]
            s1.append(LU1[dim-1][i_spec].solve(s1_arg))
            s2_arg = 8*c3[i_spec]+(3/2)*dt*c4[i_spec]+0.5*dt*F_star[i_spec]
            s2.append(LU2[dim-1][i_spec].solve(s2_arg))

        # Compute final value of U in equation (19)
        u_old = [x-y for (x, y) in zip(s1, s2)]
        if save_all_steps:
            for i_spec in range(num_species):
                all_steps[i, i_spec] = u_old[i_spec]

    # %     % For u
    # %     b1 = U1{1}\(L1{1}\F_old(:,1));
    # %     b2 = U2{1}\(L2{1}\F_old(:,1));
    # %     c2 = 9*b1-8*b2;
    # %     b3 = U1{2}\(L1{2}\c2);
    # %     b4 = U2{2}\(L2{2}\c2);
    # %     c4u = 9*b3-8*b4;
    # %     % For v
    # %     b1 = U1{1}\(L1{1}\F_old(:,2));
    # %     b2 = U2{1}\(L2{1}\F_old(:,2));
    # %     c2 = 9*b1-8*b2;
    # %     b3 = U1{2}\(L1{2}\c2);
    # %     b4 = U2{2}\(L2{2}\c2);
    # %     c4v = 9*b3-8*b4;
    # %
    # %     %For u
    # %     a1 = U1{1}\(L1{1}\u_old);
    # %     a2 = U2{1}\(L2{1}\u_old);
    # %     c1 = 9*a1-8*a2;
    # %     a3 = U1{2}\(L1{2}\c1);
    # %     a4 = U2{2}\(L2{2}\c1);
    # %     c3u = 9*a3-8*a4;
    # %     s1u = U1{3}\(L1{3}\(9*c3u+2*dt*c4u+dt*F_star(:,1)));
    # %     s2u = U2{3}\(L2{3}\(8*c3u+(3/2)*dt*c4u+0.5*dt*F_star(:,1)));
    # %     u_old = s1u-s2u;
    # %     % For v
    # %     a1 = U1{1}\(L1{1}\v_old);
    # %     a2 = U2{1}\(L2{1}\v_old);
    # %     c1 = 9*a1-8*a2;
    # %     a3 = U1{2}\(L1{2}\c1);
    # %     a4 = U2{2}\(L2{2}\c1);
    # %     c3v = 9*a3-8*a4;
    # %     s1v = U1{3}\(L1{3}\(9*c3v+2*dt*c4v+dt*F_star(:,2)));
    # %     s2v = U2{3}\(L2{3}\(8*c3v+(3/2)*dt*c4v+0.5*dt*F_star(:,2)));
    # %     v_old = s1v-s2v;
    u_soln = u_old
    runtime = time.time() - start_time

    if save_all_steps:
        return runtime, all_steps
    else:
        return runtime, u_soln