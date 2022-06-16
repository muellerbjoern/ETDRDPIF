import numpy as np
import scipy.sparse as sp


def discretize_periodic(steps, square_len, Diff, Adv):

    dim = Adv.shape[1]

    # create nodes
    # Split interval into steps subintervals (i.e., steps+1 points, including the
    # end of the interval
    x = np.linspace(0,square_len,steps+1)
    h = abs(x[0]-x[1])

    # Remove the very last point, i.e., the end of the interval, as it is the same
    # as the very first (periodic boundary!)
    x = x[:-1]

    # Create nodes using ndgrid
    # This is limited to a (hyper)square domain currently
    # For other (hyper)rectangles, explicitly specify grid vectors for all
    # dimensions
    nodes = np.array(np.meshgrid(*([x]*dim), indexing='ij'))  # TODO: Check if this is really desired
    nodes = nodes.reshape(dim, -1).T
    nodes = nodes [:, ::-1]  # TODO: In Matlab this is the result, but do I want that in Python?

    # Block matrix Assembly
    # Matrix for the 1D, 1 species case
    e = np.ones(steps)
    r = 1/h**2
    B = sp.spdiags([-r*e, 2*r*e, -r*e], [-1, 0, 1], steps, steps, format='csc')

    # Periodic boundary condition
    B[0, -1] = -r
    B[-1, 0] = -r

    # Advection matrix analogously
    r_adv = 1 / (2 * h)
    C = -sp.spdiags([-r_adv * e, r_adv * e], [-1,1], steps, steps, format='csc')
    C[0, -1] = r_adv
    C[-1, 0] = -r_adv

    A = adapt_dimension(B, C, Diff, Adv)

    return x, steps, nodes, A


def adapt_dimension(B, C, Diff, Adv):
    # Diff and Adv must be (num_species x dim) matrices.
    # This enables setting diffusion and advection per species and
    # per dimension.
    # Dependence on spatial dimension is mostly required for advection as
    # advection is usually defined along a certain spatial vector
    # Different diffusion constants per species necessitate the ability
    # of setting constants per species
    # In order to keep it general, we allow both dependencies

    num_species, dim = Diff.shape
    if Diff.shape != Adv.shape:
        msg = 'Advection and Diffusion constant matrices need to be '\
            '(number of species x spatial dimension)'
        raise ValueError(msg)

    steps = B.shape[0]
    if B.shape != C.shape:
        msg = 'Advection and Diffusion matrices (spatial discretization)'\
            ' need to be (steps x steps)'
        raise ValueError(msg)

    A = []  # dimensions x species  # TODO: NOTE! Different from Matlab

    I = sp.eye(steps, format='csc')

    for i_dim in range(1, dim+1):
        I_left = 1
        for ii in range(i_dim+1, dim+1):
            I_left = sp.kron(I_left, I)

        I_right = 1
        for ii in range(2, i_dim+1):
           I_right = sp.kron(I_right, I)

        A.append([])
        for i_spec in range(num_species):
            A[i_dim-1].append(Diff[i_spec,i_dim-1]*sp.kron(I_left, sp.kron(B, I_right)) +  Adv[i_spec,i_dim-1]*sp.kron(I_left, sp.kron(C, I_right)))

    return A

    # 3D equivalent:
    # Ax = Diff(1,1)*kron(I,kron(I,B)) + Adv(1,1)*kron(I,kron(I,C));
    # Ay = Diff(1,1)*kron(I,kron(B,I)) + Adv(1,1)*kron(I,kron(C,I));
    # Az = Diff(1,1)*kron(B,kron(I,I)) + Adv(1,1)*kron(C,kron(I,I));


if __name__ == "__main__":
    A = discretize_periodic(10, 1, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    print(A)
