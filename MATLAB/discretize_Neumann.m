function [x, nodes, B, C] = discretize_Neumann(steps, square_len, dim)

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
    C(1, 2) = 0;
    C(steps, steps-1) = 0;
    
end
