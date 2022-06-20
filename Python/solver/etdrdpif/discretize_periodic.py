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
    B = sp.spdiags([-r*e, 2*r*e, -r*e], [-1, 0, 1], steps, steps, format='lil')

    # Periodic boundary condition
    B[0, -1] = -r
    B[-1, 0] = -r

    # Advection matrix analogously
    r_adv = 1 / (2 * h)
    C = -sp.spdiags([-r_adv * e, r_adv * e], [-1,1], steps, steps, format='lil')
    C[0, -1] = r_adv
    C[-1, 0] = -r_adv

    A = adapt_dimension(B, C, Diff, Adv)

    return x, steps, nodes, A


def discretize_upwind_periodic(steps, square_len, Diff, Adv):
    num_species, dim = Adv.shape

    # create nodes
    # Split interval into steps subintervals (i.e., steps+1 points, including the
    # end of the interval
    x = np.linspace(0, square_len, steps + 1)
    h = abs(x[0] - x[1])

    # Remove the very last point, i.e., the end of the interval, as it is the same
    # as the very first (periodic boundary!)
    x = x[:-1]

    # Create nodes using ndgrid
    # This is limited to a (hyper)square domain currently
    # For other (hyper)rectangles, explicitly specify grid vectors for all
    # dimensions
    nodes = np.array(np.meshgrid(*([x] * dim), indexing='ij'))  # TODO: Check if this is really desired
    nodes = nodes.reshape(dim, -1).T
    nodes = nodes[:, ::-1]  # TODO: In Matlab this is the result, but do I want that in Python?

    # Block matrix Assembly
    # Matrix for the 1D, 1 species case
    e = np.ones(steps)
    r = 1 / h ** 2
    B = sp.spdiags([-r * e, 2 * r * e, -r * e], [-1, 0, 1], steps, steps, format='lil')

    # Periodic boundary condition
    B[0, -1] = -r
    B[-1, 0] = -r

    # Advection matrix analogously
    r_adv = 1 / h
    Cs = []
    for i_dim in range(dim):
        Cs.append([])
        for i_spec in range(num_species):
            if Adv[i_spec][i_dim] >= 0:
                C = sp.spdiags([r_adv * e, -r_adv * e], [0, 1], steps, steps, format='lil')
                # C[0, -1] = -r_adv
                C[-1, 0] = -r_adv
            else:
                C = sp.spdiags([r_adv * e, -r_adv * e], [-1, 0], steps, steps, format='lil')
                C[0, -1] = r_adv
                # C[-1, 0] = r_adv

            Cs[i_dim].append(C)

    A = adapt_dimension(B, Cs, Diff, Adv)

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

    """
    params: C list[list[scipy.sparse.matrix]] or scipy.sparse.matrix # TODO: what is the matrix type?
    """

    num_species, dim = Diff.shape
    if Diff.shape != Adv.shape:
        msg = 'Advection and Diffusion constant matrices need to be ' \
              '(number of species x spatial dimension)'
        raise ValueError(msg)

    if isinstance(C, list):
        C0 = C[0][0]
    else:
        C0 = C
    steps = B.shape[0]
    if B.shape != C0.shape:
        msg = 'Advection and Diffusion matrices (spatial discretization)' \
              ' need to be (steps x steps)'
        raise ValueError(msg)

    A = []  # dimensions x species  # TODO: NOTE! Different from Matlab

    I = sp.eye(steps, format='csc')

    for i_dim in range(1, dim + 1):
        I_left = 1
        for ii in range(i_dim + 1, dim + 1):
            I_left = sp.kron(I_left, I)

        I_right = 1
        for ii in range(2, i_dim + 1):
            I_right = sp.kron(I_right, I)

        A.append([])
        for i_spec in range(num_species):
            if isinstance(C, list):
                curr_C = C[i_dim - 1][i_spec]
            else:
                curr_C = C
            A[i_dim - 1].append(
                Diff[i_spec, i_dim - 1] * sp.kron(I_left, sp.kron(B, I_right)) + Adv[i_spec, i_dim - 1] * sp.kron(
                    I_left, sp.kron(curr_C, I_right)))

    return A

    # 3D equivalent:
    # Ax = Diff(1,1)*kron(I,kron(I,B)) + Adv(1,1)*kron(I,kron(I,C));
    # Ay = Diff(1,1)*kron(I,kron(B,I)) + Adv(1,1)*kron(I,kron(C,I));
    # Az = Diff(1,1)*kron(B,kron(I,I)) + Adv(1,1)*kron(C,kron(I,I));


if __name__ == "__main__":
    A = discretize_periodic(10, 1, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    print(A)
