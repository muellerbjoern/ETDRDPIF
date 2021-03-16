% ******fprintf('This program was developed by Harish Bhatt)***************
% ************Brusselator 3D model with no flux boundary condition*********
% clc;
% %clear all;
% format long
%**************************************************************************
%order = zeros(1,3); Error = zeros(1,4);CPUTIME = zeros(1,5);
%for ii = 0:1
%**********************************Inputs**********************************
function [time,u_soln] = Krylov_Periodic_benchmark(te, dt,steps)
    x0 = 0; xn = 2*pi; t0 = 0; tn = te; d1 = 1.0/3.0;d2 = 1.0/3.0;
    a = 1;a1 = 1;
    b = 100.0;
    c = 1.0;
    %nn = steps;
    k = dt;% time step
    %x = x0:h:xn; % space discretization
    x = linspace(x0,xn,steps+1); h = abs(x(1)-x(2));
    
    disp(x);
    disp(size(x));
    y = x;
    z = y;
    t = t0:k:tn; % time discretization
    n = length(x);   %n points along x direction
    m = length(y);  %m points along y direction
    K = length(t);  %k points along time direction
    %**************************construction of band matrix*****************
    e = ones(n,1);
    B = spdiags([ (2*d1+a1*h)*e -4*d1*e (2*d1-a1*h)*e],-1:1,n,n);%tridiagonal matrix
    B(1,n) = (2*d1+a1*h);
    B(n,1) = (2*d1-a1*h);
    B1 = spdiags([ (2*d2+a1*h)*e -4*d2*e (2*d2-a1*h)*e],-1:1,n,n);%tridiagonal matrix
    B1(1,n) = (2*d2+a1*h);
    B1(n,1) = (2*d2-a1*h);
    A1 = 1/(2*h^2)*( kron(B, kron(speye(n), speye(n))) + kron(speye(n),...
        kron(B, speye(n)))+ kron(speye(n), kron(speye(n),B)));
    A2 = 1/(2*h^2)*( kron(B1, kron(speye(n), speye(n))) + kron(speye(n),...
        kron(B1, speye(n)))+ kron(speye(n), kron(speye(n),B1)));
    % %********************************************************************
    U = zeros(n,n,n);
    V = zeros(n,n,n);
    % initial condition set up
     for p = 1:n
            for q = 1:m
                for i = 1:m
                    U(p, q, i) = 2.0*cos(x(p) + y(q) + z(i));
                    V(p, q, i) = (b-c)*cos(x(p) + y(q) + z(i));
                end
            end
     end
    U_1 = U(:); V_1 = V(:);
    U_2 = U_1; V_2 = V_1;
    U_3 = U_1; V_3 = V_1; % nonlinear function setup
    F = @(U1,V1) U1.^2.*V1-(a+1)*U1+b;
    G = @(U1,V1) a*U1-U1.^2.*V1;
    T = t0:2*k:tn; M1 = length(T);
    m1 = 10; % Krylov subspace dimension
    tic;
for l = 2:M1
    disp(l);
    U_1 = expv( k, A1, ( U_1 + k*F(U_1, V_1)), 1.0e-1, m1);
    V_1 = expv( k, A2, ( V_1 + k*G(U_1,V_1)),1.0e-1,m1);
    
    U_1 = expv( k, A1, (U_1 + k*F(U_1, V_1)),1.0e-1,m1);
    V_1 = expv( k, A2, ( V_1 + k*G(U_1,V_1)),1.0e-1,m1);
    
%   Extrapolation Scheme 
    
    U_2 = expv( 2*k, A1, (U_2 + 2*k*F(U_2, V_2)),1.0e-1,m1);
    V_2 = expv( 2*k, A2, (V_2 + 2*k*G(U_2,V_2)),1.0e-1,m1);
    
    U_3 = expv( 2*k, A1, (U_3 + 2*k*F(U_3, V_3)),1.0e-1,m1);
    V_3 = expv( 2*k, A2, (V_3 + 2*k*G(U_3,V_3)),1.0e-1,m1);
    
    sol1 = 2*U_1-(U_2+U_3)/2;
    sol2 = 2*V_1-(V_2+V_3)/2;   % Extrapolation scheme
    
    U_1 = sol1;V_1 = sol2;U_2 = U_1;
    V_2 = V_1;U_3 = U_1; V_3 = V_1;
    
end
time = toc;
u_soln = sol1;


%     CPUTIME(ii+1) = toc;
%     U = reshape(sol1,[n n,n]);
%     V = reshape(sol2,[n n,n]);
%     if ii >=1
%         error = norm(U(:)-U_n(:),inf);% maximum Error
%     %     error1 = sqrt(h*temp1*temp1')% L2 error norm
%         if ii >=2
%             order(ii) = (log10(error2)-log10(error))/log10(2.0);
%         end
%         Error(ii+1) = error;
%         error2=error;
%     end
%     U_n = U;
% end
% % ******************************output*************************************
% Error, order, CPUTIME