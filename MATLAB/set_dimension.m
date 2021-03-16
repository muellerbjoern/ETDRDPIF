function A = set_dimension(B, C, Diff, Adv)

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