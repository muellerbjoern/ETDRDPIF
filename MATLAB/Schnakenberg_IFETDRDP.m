function [runtime,u_soln,u_ex] = Schnakenberg_IFETDRDP(dt,steps,do_plot)

% dt: time step. Default is 0.001
% steps: number of spatial points in each coordinate direction. Default is 11

%# k is temporal discretization (dt); here: 0.005
%# h is spatial discretization (steps); here: 0.1

te = 1.0;
square_len = 2*pi;


%% Model Paramters and initial conditions
a = 3.0; 
%#d = 0.1;
d=1.0;
Adv = a/3.0*eye(2);
Diff = d/3.0*eye(2);

%#b = 0.1; 
b = 100.0;
c = 1.0;

% create nodes
%# Split interval into steps subintervals (i.e., steps+1 points, including the
%# end of the interval
x = linspace(0,square_len,steps+1); h = abs(x(1)-x(2)); 
%# Remove the very last point, i.e., the end of the interval, as it is the same
%# as the very first (periodic boundary!)
x = x(1:steps);
y = x;
z = y;
nnodes = steps^3;
nodes = zeros(nnodes,3);
m = 1;
for k = 1:steps
    for j = 1 : steps
            for i = 1:steps
                   nodes(m,:) = [x(i) y(j) z(k)];
                m = m+1;
            end
     end
end

% discretize time interval
t = 0:dt:te; tlen = length(t);

%# Both species treated separately!
%# Possible due to assumption of no coupling in diffusive term
% initial condition for u
%#u_old = 1 + sin(2*pi*nodes(:,1)).*sin(2*pi*nodes(:,2)).*sin(2*pi*nodes(:,3));
u_old = 2*cos(sum(nodes, 2));

% initial condition for v
%#v_old = 3*ones(nnodes,1); 
v_old = (b-c)*cos(sum(nodes,2));

%% Block matrix Assembly
% 1D  matrix
I = speye(steps);
e = ones(steps,1);r=1/h^2;
B = spdiags([-r*e 2*r*e -r*e], -1:1, steps, steps);
%#B(1,2) = -2*r;
%#B(steps,steps-1) = -2*r;
%# Periodic boundary condition
B(1, steps) = -r;
B(steps, 1) = -r;

%# Advection matrix analogously
r_adv = 1/(2*h);
C = spdiags([-r_adv*e 0*e r_adv*e], -1:1, steps, steps);
C(1, steps) = -r_adv;
C(steps, 1) = r_adv;

[runtime, u_soln, v_soln] = solve_ETD(dt, tlen, steps, 3, 2, B, C, Diff, Adv, u_old, v_old, @F);

  Uex = (exp(-b-d)+exp(-c-d))*cos(sum(nodes, 2)-a);
  Vex = (b-c)*exp(-c-d)*cos(sum(nodes, 2)-a);
  Uex = reshape(Uex, steps, steps, steps);
  Vex = reshape(Vex, steps, steps, steps);
  u_ex = Uex;
  if do_plot
  Usoln = reshape(u_soln,steps,steps,steps); 
  Vsoln = reshape(v_soln,steps,steps,steps);
  Uplot = Usoln(:,:,steps);
  Vplot = Vsoln(:,:,steps);

    figure(15)
    contourf(x,y,Uplot')
    xlabel('x')
    ylabel('y')
    title("U")
    colorbar
    set(gca,'LineWidth', 1);
    set(gca,'FontSize',10);
    set(gca,'FontWeight','bold');
    pbaspect(gca,[1 1 1])
    %#set(gca,'XTick',[0 0.2 0.4 0.6 0.8 1]);
    %#set(gca,'YTick',[0 0.2 0.4 0.6 0.8 1]);
    %#print -depsc2 sliceu.eps
    
    figure(16)
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
    
    

  Uexplot = Uex(:,:,steps);
  Vexplot = Vex(:,:,steps);
  
  disp(max(max(max(Usoln - Uex))));

    figure(17)
    contourf(x,y,Uexplot')
    xlabel('x')
    ylabel('y')
    title("U exact")
    colorbar
    set(gca,'LineWidth', 1);
    set(gca,'FontSize',10);
    set(gca,'FontWeight','bold');
    pbaspect(gca,[1 1 1])
    %#set(gca,'XTick',[0 0.2 0.4 0.6 0.8 1]);
    %#set(gca,'YTick',[0 0.2 0.4 0.6 0.8 1]);
    %#print -depsc2 sliceu.eps
    
    figure(18)
    contourf(x,y,Vexplot')
    xlabel('x')
    ylabel('y')
    title("V exact")
    colorbar
    set(gca,'LineWidth', 1);
    set(gca,'FontSize',10);
    set(gca,'FontWeight','bold');
    pbaspect(gca,[1 1 1])
    %#set(gca,'XTick',[0 0.2 0.4 0.6 0.8 1]);
    %#set(gca,'YTick',[0 0.2 0.4 0.6 0.8 1]);
    %#print -depsc2 slicev.eps
    
   figure(1)

   plot(x, Uex(:, 2, 2))
      hold on
   plot(x, Usoln(:, 2, 2),'o')
   title("Krylov Fig. 3a")
   hold off
   shg
   
      figure(2)

   plot(x, Vex(:, 2, 2))
      hold on
   plot(x, Vsoln(:, 2, 2),'o')
   title("Krylov Fig. 3b")
   hold off
   shg

   
   figure(19)
    contourf(x,y,Uexplot-Uplot')
    xlabel('x')
    ylabel('y')
    colorbar
    set(gca,'LineWidth', 1);
    set(gca,'FontSize',10);
    set(gca,'FontWeight','bold');
    
    title("U error")
    
    pbaspect(gca,[1 1 1])
    
  end
    

function Fr = F(u,v)
 f1 = -b*u + v;
 f2 = -c*v;
 Fr = [f1 f2];
end

end

function [runtime, u_soln, v_soln] = solve_ETD(dt, tlen, steps, dim, num_species, B, C, Diff, Adv, u_old, v_old, F)

%# TODO: Different matrices for different species!
%# Currently treating diffusion and advection as constant across species
% 3D  matrix for space

%A = sparse(steps^dim, dim);

A = cell(dim, 1);

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
    A{i_dim} = Diff(1,1)*kron(I_left, kron(B, I_right)) +  Adv(1,1)*kron(I_left, kron(C, I_right));
end

Ax = Diff(1,1)*kron(I,kron(I,B)) + Adv(1,1)*kron(I,kron(I,C));
Ay = Diff(1,1)*kron(I,kron(B,I)) + Adv(1,1)*kron(I,kron(C,I));
Az = Diff(1,1)*kron(B,kron(I,I)) + Adv(1,1)*kron(C,kron(I,I));

%#figure(100);
%#spy(Ax);
%#figure(101);
%#spy(Ay);
%#figure(102);
%#spy(Az);

%#Ax = Adv(1,1)*kron(I,kron(I,C));
%#Ay = Adv(1,1)*kron(I,kron(C,I));
%#Az = Adv(1,1)*kron(C,kron(I,I));

% System matrices
r1 = 1/3; r2 = 1/4;
% Identity matrix of correct size
Id_temp = 1;
for i_dim = 1:dim
   Id_temp = kron(Id_temp, I);
end
% Stack identity matrices analogously to A

Id = cell(dim, 1);
for i_dim = 1:dim
    Id{i_dim} = Id_temp;
end

% Cellfun applies function to all arrays stored in cell array
A1 = cellfun(@plus, Id, cellfun(@(x) x*r1*dt, A, 'UniformOutput', false), 'UniformOutput', false);
A2 = cellfun(@plus, Id, cellfun(@(x) x*r2*dt, A, 'UniformOutput', false), 'UniformOutput', false);
A3 = cellfun(@plus, Id, cellfun(@(x) x*dt, A, 'UniformOutput', false), 'UniformOutput', false);


A1x = (Id_temp + r1*dt*Ax);
A2x = (Id_temp + r2*dt*Ax);
A3x = (Id_temp + dt*Ax);
A1y = (Id_temp + r1*dt*Ay);
A2y = (Id_temp + r2*dt*Ay);
A3y = (Id_temp + dt*Ay);
A1z = (Id_temp + r1*dt*Az);
A2z = (Id_temp + r2*dt*Az);
A3z = (Id_temp + dt*Az);



clear Ax Ay Az I Id B

L1 = cell(dim, 1);
U1 = cell(dim, 1);
L2 = cell(dim, 1);
U2 = cell(dim, 1);
L3 = cell(dim, 1);
U3 = cell(dim, 1);

for i_dim = 1:dim
   [L1{i_dim}, U1{i_dim}] = lu(A1{i_dim});
   [L2{i_dim}, U2{i_dim}] = lu(A2{i_dim});
   [L3{i_dim}, U3{i_dim}] = lu(A3{i_dim});
end

[L3x,U3x]=lu(A3x);
[L3y,U3y]=lu(A3y);
[L3z,U3z]=lu(A3z);

[L2x,U2x]=lu(A2x);
[L2y,U2y]=lu(A2y);
[L2z,U2z]=lu(A2z);

[L1x,U1x]=lu(A1x);
[L1y,U1y]=lu(A1y);
[L1z,U1z]=lu(A1z);

% figure(1);
% spy(L3{3});
% figure(2);
% spy(U3{3});

tic
for i = 2:tlen
  
    %#disp(i);
     
    F_old = F(u_old,v_old);
    
    % TODO: Replace by more general formulation
    all_u_old = cell(2);
    all_u_old{1} = u_old;
    all_u_old{2} = v_old;
    
    p = cell(num_species);
    my_d = cell(num_species);
    for i_spec = 1:num_species
        p{i_spec} = U3{1}\(L3{1}\F_old(:, i_spec));
        my_d{i_spec} = U3{1}\(L3{1}\all_u_old{i_spec});
        for i_dim = 2:dim
            p{i_spec} = U3{i_dim}\(L3{i_dim}\p{i_spec});
            my_d{i_spec} = U3{i_dim}\(L3{i_dim}\my_d{i_spec});
        end
    end
    
    all_u_star = cellfun(@plus, my_d, cellfun(@(x) x*dt, p, 'UniformOutput', false), 'UniformOutput', false);
    F_star = F(all_u_star{1}, all_u_star{2});
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
       
    % For u
    b1 = U1{1}\(L1{1}\F_old(:,1));
    b2 = U2{1}\(L2{1}\F_old(:,1));
    c2 = 9*b1-8*b2;
    b3 = U1{2}\(L1{2}\c2);
    b4 = U2{2}\(L2{2}\c2);
    c4u = 9*b3-8*b4;
    % For v
    b1 = U1{1}\(L1{1}\F_old(:,2));
    b2 = U2{1}\(L2{1}\F_old(:,2));
    c2 = 9*b1-8*b2;
    b3 = U1{2}\(L1{2}\c2);
    b4 = U2{2}\(L2{2}\c2);
    c4v = 9*b3-8*b4;
    
    % For u
    a1 = U1{1}\(L1{1}\u_old);
    a2 = U2{1}\(L2{1}\u_old);
    c1 = 9*a1-8*a2;
    a3 = U1{2}\(L1{2}\c1);
    a4 = U2{2}\(L2{2}\c1);
    c3u = 9*a3-8*a4;
    s1u = U1{3}\(L1{3}\(9*c3u+2*dt*c4u+dt*F_star(:,1)));
    s2u = U2{3}\(L2{3}\(8*c3u+(3/2)*dt*c4u+0.5*dt*F_star(:,1)));
    u_old = s1u-s2u;
    % For v
    a1 = U1{1}\(L1{1}\v_old);
    a2 = U2{1}\(L2{1}\v_old);
    c1 = 9*a1-8*a2;
    a3 = U1{2}\(L1{2}\c1);
    a4 = U2{2}\(L2{2}\c1);
    c3v = 9*a3-8*a4;
    s1v = U1{3}\(L1{3}\(9*c3v+2*dt*c4v+dt*F_star(:,2)));
    s2v = U2{3}\(L2{3}\(8*c3v+(3/2)*dt*c4v+0.5*dt*F_star(:,2)));
    v_old = s1v-s2v;
end
u_soln = u_old;
v_soln = v_old;
runtime = toc;

end
