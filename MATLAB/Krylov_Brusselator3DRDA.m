% ******fprintf('This program was developed by Harish Bhatt)***************
% ************Brusselator 3D model with no flux boundary condition*********
% clc;
% %clear all;
% format long
%**************************************************************************
%order = zeros(1,3); Error = zeros(1,4);CPUTIME = zeros(1,5);
%for ii = 0:1
%**********************************Inputs**********************************
function [time,u_soln] = Krylov_Brusselator3DRDA(te, dt,steps)
    x0 = 0; xn = 1; t0 = 0; tn = te; d1 = 0.02;d2 = 0.01;
    a = 1; b = 2;a1 = 1;
    nn = steps; h = 1/nn;
    k = dt;% time step
    x = x0:h:xn; % space discretization
    y = x;
    z = y;
    t = t0:k:tn; % time discretization
    n = length(x);   %n points along x direction
    m = length(y);  %m points along y direction
    K = length(t);  %k points along time direction
    %**************************construction of band matrix*****************
    e = ones(n,1);
    B = spdiags([ (2*d1+a1*h)*e -4*d1*e (2*d1-a1*h)*e],-1:1,n,n);%tridiagonal matrix
    B(1,2) = 2*(2*d1);
    B(n,n-1) = 2*(2*d1);
    B1 = spdiags([ (2*d2+a1*h)*e -4*d2*e (2*d2-a1*h)*e],-1:1,n,n);%tridiagonal matrix
    B1(1,2) = 2*(2*d2);
    B1(n,n-1) = 2*(2*d2);
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
                    U(p,q,i) = 1.0+sin(2*pi*x(p))*sin(2*pi*y(q))*sin(2*pi*z(i));
                    V(p,q,i) = 3.0;
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
    tol = 1.0e-7;
    U_1 = expv( k, A1, ( U_1 + k*F(U_1, V_1)),tol,m1);
    V_1 = expv( k, A2, ( V_1 + k*G(U_1,V_1)),tol,m1);
    
    U_1 = expv( k, A1, (U_1 + k*F(U_1, V_1)),tol,m1);
    V_1 = expv( k, A2, ( V_1 + k*G(U_1,V_1)),tol,m1);
    
%   Extrapolation Scheme 
    
    U_2 = expv( 2*k, A1, (U_2 + 2*k*F(U_2, V_2)),tol,m1);
    V_2 = expv( 2*k, A2, (V_2 + 2*k*G(U_2,V_2)),tol,m1);
    
    U_3 = expv( 2*k, A1, (U_3 + 2*k*F(U_3, V_3)),tol,m1);
    V_3 = expv( 2*k, A2, (V_3 + 2*k*G(U_3,V_3)),tol,m1);
    
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