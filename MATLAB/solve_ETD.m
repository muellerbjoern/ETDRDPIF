function [runtime, u_soln] = solve_ETD(dt, tlen, steps, A, u_old, F)

[num_species, dim] = size(A);

% System matrices
r1 = 1/3; r2 = 1/4;
% Identity matrix of correct size
Id_temp = 1;
I = speye(steps);
for i_dim = 1:dim
   Id_temp = kron(Id_temp, I);
end
% Stack identity matrices analogously to A

Id = cell(num_species, dim);
for i_dim = 1:dim
    for i_spec = 1:num_species
        Id{i_spec, i_dim} = Id_temp;
    end
end

% Cellfun applies function to all arrays stored in cell array
A1 = cellfun(@plus, Id, cellfun(@(x) x*r1*dt, A, 'UniformOutput', false), 'UniformOutput', false);
A2 = cellfun(@plus, Id, cellfun(@(x) x*r2*dt, A, 'UniformOutput', false), 'UniformOutput', false);
A3 = cellfun(@plus, Id, cellfun(@(x) x*dt, A, 'UniformOutput', false), 'UniformOutput', false);

% 3D equivalent
% A1x = (Id_temp + r1*dt*Ax);
% A2x = (Id_temp + r2*dt*Ax);
% A3x = (Id_temp + dt*Ax);
% A1y = (Id_temp + r1*dt*Ay);
% A2y = (Id_temp + r2*dt*Ay);
% A3y = (Id_temp + dt*Ay);
% A1z = (Id_temp + r1*dt*Az);
% A2z = (Id_temp + r2*dt*Az);
% A3z = (Id_temp + dt*Az);

clear A Id_temp Id B C I

% Dimensionwise LU decomposition
[L1, U1] = cellfun(@lu, A1, 'UniformOutput', false);
[L2, U2] = cellfun(@lu, A2, 'UniformOutput', false);
[L3, U3] = cellfun(@lu, A3, 'UniformOutput', false);
% 3D equivalent:
% [L3x,U3x]=lu(A3x);
% [L3y,U3y]=lu(A3y);
% [L3z,U3z]=lu(A3z);
% 
% [L2x,U2x]=lu(A2x);
% [L2y,U2y]=lu(A2y);
% [L2z,U2z]=lu(A2z);
% 
% [L1x,U1x]=lu(A1x);
% [L1y,U1y]=lu(A1y);
% [L1z,U1z]=lu(A1z);

clear A1 A2 A3

tic
for i = 2:tlen
     
    F_old = F(u_old);
    
    p = cell(num_species, 1);
    d = cell(num_species, 1);
    for i_spec = 1:num_species
        p{i_spec} = F_old(:, i_spec);
        d{i_spec} = u_old{i_spec};
        for i_dim = 1:dim
            %# TODO: Aggregate RHS, might be faster due to BLAS routine?
            %p{i_spec} = U3{i_spec, i_dim}\(L3{i_spec, i_dim}\p{i_spec});
            %d{i_spec} = U3{i_spec, i_dim}\(L3{i_spec, i_dim}\d{i_spec});
            p_d = U3{i_spec, i_dim}\(L3{i_spec, i_dim}\[p{i_spec} d{i_spec}]);
            p{i_spec} = p_d(:, 1);
            d{i_spec} = p_d(:, 2);
        end
    end
    u_star = cellfun(@plus, d, cellfun(@(x) x*dt, p, 'UniformOutput', false), 'UniformOutput', false);
    F_star = F(u_star);
%     % For u
%     p1 = U3x\(L3x\F_old(:,1));
%     p2 = U3y\(L3y\p1);
%     p3u = U3z\(L3z\p2);  
%     % For v
%     p1 = U3x\(L3x\F_old(:,2));
%     p2 = U3y\(L3y\p1);
%     p3v = U3z\(L3z\p2);
%     
%     % For u
%     d1 = U3x\(L3x\u_old);
%     d2 = U3y\(L3y\d1);
%     d3u = U3z\(L3z\d2);
%     u_star = d3u + dt*p3u;
%     % For v
%     d1 = U3x\(L3x\v_old);
%     d2 = U3y\(L3y\d1);
%     d3v = U3z\(L3z\d2);
%     v_star = d3v + dt*p3v;
%     F_star = F(u_star,v_star);
       
    % Cell arrays to store intermediate results
    
    % Contains intermediate RHS needed to compute c4 (F_old, c2, c4)
    c4 = cell(num_species, 1);
    % Contains intermediate RHS needed to compute c3 (u_old, c1, c3)
    c3 = cell(num_species, 1);
    
    s1 = cell(num_species, 1);
    s2 = cell(num_species, 1);
    
    for i_spec = 1:num_species
        % Initialize RHS
        c4{i_spec} = F_old(:, i_spec);
        c3{i_spec} = u_old{i_spec};
        for i_dim = 1:dim-1
            % Solve for c4, linear system with F_old as RHS
            a1_b1 = U1{i_spec, i_dim}\(L1{i_spec, i_dim}\[c3{i_spec} c4{i_spec}]);
            a1 = a1_b1(:, 1);
            b1 = a1_b1(:, 2);
            a2_b2 = U2{i_spec, i_dim}\(L2{i_spec, i_dim}\[c3{i_spec} c4{i_spec}]);
            a2 = a2_b2(:, 1);
            b2 = a2_b2(:, 2);
            c4{i_spec} = 9*b1-8*b2;
            
            % Solve for c3, linear system with u_old as RHS
            %a1 = U1{i_spec, i_dim}\(L1{i_spec, i_dim}\c3{i_spec});
            %a2 = U2{i_spec, i_dim}\(L2{i_spec, i_dim}\c3{i_spec});
            c3{i_spec} = 9*a1-8*a2;
        end
        
        % Summarize c3 and c4 to the summands of equation (19)
        % (Asante-Asamani, 2020)
        s1{i_spec} = U1{i_spec, dim}\(L1{i_spec, dim}\(9*c3{i_spec}+2*dt*c4{i_spec}+dt*F_star(:,i_spec)));
        s2{i_spec} = U2{i_spec, dim}\(L2{i_spec, dim}\(8*c3{i_spec}+(3/2)*dt*c4{i_spec}+0.5*dt*F_star(:,i_spec)));
        
    end
    
    % Compute final value of U in equation (19)
    u_old = cellfun(@minus, s1, s2, 'UniformOutput', false);

%     % For u
%     b1 = U1{1}\(L1{1}\F_old(:,1));
%     b2 = U2{1}\(L2{1}\F_old(:,1));
%     c2 = 9*b1-8*b2;
%     b3 = U1{2}\(L1{2}\c2);
%     b4 = U2{2}\(L2{2}\c2);
%     c4u = 9*b3-8*b4;
%     % For v
%     b1 = U1{1}\(L1{1}\F_old(:,2));
%     b2 = U2{1}\(L2{1}\F_old(:,2));
%     c2 = 9*b1-8*b2;
%     b3 = U1{2}\(L1{2}\c2);
%     b4 = U2{2}\(L2{2}\c2);
%     c4v = 9*b3-8*b4;
%     
%     %For u
%     a1 = U1{1}\(L1{1}\u_old);
%     a2 = U2{1}\(L2{1}\u_old);
%     c1 = 9*a1-8*a2;
%     a3 = U1{2}\(L1{2}\c1);
%     a4 = U2{2}\(L2{2}\c1);
%     c3u = 9*a3-8*a4;
%     s1u = U1{3}\(L1{3}\(9*c3u+2*dt*c4u+dt*F_star(:,1)));
%     s2u = U2{3}\(L2{3}\(8*c3u+(3/2)*dt*c4u+0.5*dt*F_star(:,1)));
%     u_old = s1u-s2u;
%     % For v
%     a1 = U1{1}\(L1{1}\v_old);
%     a2 = U2{1}\(L2{1}\v_old);
%     c1 = 9*a1-8*a2;
%     a3 = U1{2}\(L1{2}\c1);
%     a4 = U2{2}\(L2{2}\c1);
%     c3v = 9*a3-8*a4;
%     s1v = U1{3}\(L1{3}\(9*c3v+2*dt*c4v+dt*F_star(:,2)));
%     s2v = U2{3}\(L2{3}\(8*c3v+(3/2)*dt*c4v+0.5*dt*F_star(:,2)));
%     v_old = s1v-s2v;
end
u_soln = u_old;
runtime = toc;

end