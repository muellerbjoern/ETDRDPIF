% ******fprintf('This program was developed by Harish Bhatt)***************
% ************Brusselator 3D model with no flux boundary condition*********
% clc;
% %clear all;
% format long
%**************************************************************************
%order = zeros(1,3); Error = zeros(1,4);CPUTIME = zeros(1,5);
%for ii = 0:1
%**********************************Inputs**********************************
function [time,u_soln] = Krylov_Schnakenberg_Neumann(te, dt,steps, do_plot)
    x0 = 0; xn = 1; t0 = 0; tn = te; d1 = 0.05;d2 = 1.0;
    a = 0.1305; b = 0.7695;a1 = 0.01; a2 = 0.01;
    gamma = 100; lambda = xn - x0;
    nn = steps-1; h = 1/nn;
    k = dt;% time step
    x = x0:h:xn; % space discretization
    % Next line necessary for periodic condition!
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
    B1 = spdiags([ (2*d2+a2*h)*e -4*d2*e (2*d2-a2*h)*e],-1:1,n,n);%tridiagonal matrix
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
                    % TODO: Try cos, because it's compatible with boundary
                    % condition
                    %U(p,q,i) = 1.0+cos(2*pi*x(p))*cos(2*pi*y(q))*cos(2*pi*z(i));
                    U(p,q,i) = a+b+1e-3*exp(-100*((x(p)-lambda/3)^2 + (y(q)-lambda/2)^2 + (z(i)-lambda/3)^2));
                    V(p,q,i) = b/((a+b)^2);
                end
            end
     end
    U_1 = U(:); V_1 = V(:);
    U_2 = U_1; V_2 = V_1;
    U_3 = U_1; V_3 = V_1; % nonlinear function setup
    F = @(U1,V1) gamma*(a-U1+U1.^2.*V1);
    G = @(U1,V1) gamma*(b-U1.^2.*V1);
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

Usoln = u_soln;
Vsoln = sol2;
if do_plot
    disp(size(Usoln));
plot_soln(reshape(Usoln, steps, steps, steps), reshape(Vsoln, steps, steps, steps), {x, x}, te);

end
end


function plot_soln(Usoln, Vsoln, grid, te)
    
    x = grid{1};
    y = grid{2};

    Uplot = Usoln(:,:,1);
    Vplot = Vsoln(:,:,end);
    
    for i_plot = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32] %[5 8 9 15 16 17]
    figure()
    % Transpose Usoln! See contourf documentation
    contourf(x,y,Usoln(:, :, i_plot)')
    xlabel('x')
    ylabel('y')
    title(["U" num2str(i_plot) num2str(te)]);
    colormap(jet(256));
    colorbar
    set(gca,'LineWidth', 1);
    set(gca,'FontSize',10);
    set(gca,'FontWeight','bold');
    pbaspect(gca,[1 1 1])
    %#set(gca,'XTick',[0 0.2 0.4 0.6 0.8 1]);
    %#set(gca,'YTick',[0 0.2 0.4 0.6 0.8 1]);
    %#print -depsc2 sliceu.eps
    end
    
    figure(50)
    contourf(x,y,Vplot')
    xlabel('x')
    ylabel('y')
    title("V")
    colorbar
    set(gca,'LineWidth', 1);
    set(gca,'FontSize',10);
    set(gca,'FontWeight','bold');
    pbaspect(gca,[1 1 1])
    %#set(gca,'XTick',[0 0.2 0.4 0.6 0.8 1]);
    %#set(gca,'YTick',[0 0.2 0.4 0.6 0.8 1]);
    %#print -depsc2 slicev.eps
    
    


end


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