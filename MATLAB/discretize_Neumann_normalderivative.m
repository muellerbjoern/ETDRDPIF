function [x, steps, nodes, A] = discretize_Neumann_normalderivative(steps, square_len, Diff, Adv)

    [num_species, dim] = size(Adv);

    % create nodes
    %# Split interval into steps subintervals (i.e., steps+1 points, including the
    %# end of the interval
    x = linspace(0,square_len,steps+1); h = abs(x(1)-x(2)); 

    % Create nodes using ndgrid
    % This is limited to a (hyper)square domain currently
    % For other (hyper)rectangles, explicitly specify grid vectors for all
    % dimensions
    [nodes_cell{1:dim}] = ndgrid(x);
    nodes_cell = cellfun(@(x) reshape(x,[], 1), nodes_cell, 'UniformOutput', false);
    nodes = cell2mat(nodes_cell);

    steps = steps + 1;
    %% Block matrix Assembly
    % 1D  matrix
    e = ones(steps,1);r=1/h^2;
    B = spdiags([-r*e 2*r*e -r*e], -1:1, steps, steps);
    B(1,2) = -2*r;
    B(steps,steps-1) = -2*r;

    %# Advection matrix analogously
    r_adv = 1/(2*h);
    C = spdiags([-r_adv*e 0*e r_adv*e], -1:1, steps, steps);
    % Homogeneous Neumann boundary is worked into B using ghost points
    % Here: homogeneous Neumann interpreted as du/dn = 0
    % (n being a non-zero normal vector to the boundary)
    % Note: Different values make very little difference => What to do?
    C(1, 2) = 0;
    C(steps, steps-1) = 0;
    
    % Diff and Adv must be (num_species x dim) matrices.
% This enables setting diffusion and advection per species and
% per dimension.
% Dependence on spatial dimension is mostly required for advection as 
% advection is usually defined along a certain spatial vector
% Different diffusion constants per species necessitate the ability
% of setting constants per species
% In order to keep it general, we allow both dependencies

[num_species, dim] = size(Diff);
if size(Diff) ~= size(Adv)
    msg = ['Advection and Diffusion constant matrices need to be '...
        '(number of species x spatial dimension)'];
    error(msg);
end

steps = size(B, 1);
if size(B) ~= size(C)
    msg = ['Advection and Diffusion matrices (spatial discretization)'...
        ' need to be (steps x steps)'];
    error(msg);
end

A = cell(num_species, dim);

I = speye(steps);

for i_dim = 1:dim
    I_left = 1;
    for ii = i_dim+1:dim
        I_left = kron(I_left, I);
    end
    I_right = 1;
    for ii = 2:i_dim
       I_right = kron(I_right, I); 
    end
    for i_spec = 1:num_species
        A{i_spec, i_dim} = Diff(i_spec,i_dim)*kron(I_left, kron(B, I_right)) +  Adv(i_spec,i_dim)*kron(I_left, kron(C, I_right));
    end
end

% 3D equivalent:
% Ax = Diff(1,1)*kron(I,kron(I,B)) + Adv(1,1)*kron(I,kron(I,C));
% Ay = Diff(1,1)*kron(I,kron(B,I)) + Adv(1,1)*kron(I,kron(C,I));
% Az = Diff(1,1)*kron(B,kron(I,I)) + Adv(1,1)*kron(C,kron(I,I));
    
end
