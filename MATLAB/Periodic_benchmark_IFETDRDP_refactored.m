function [runtime,u_soln,u_ex] = Periodic_benchmark_IFETDRDP_refactored(dt,steps,do_plot)

% dt: time step. Default is 0.001
% steps: number of spatial points in each coordinate direction. Default is 11

%# k is temporal discretization (dt); here: 0.005
%# h is spatial discretization (steps); here: 0.1

dim = 3;
num_species = 2;

te = 1.0;
square_len = 2*pi;


%% Model Paramters and initial conditions
a = 3.0; 
%#d = 0.1;
d=1.0;
Adv = a/3.0*ones(num_species, dim);
Diff = d/3.0*ones(num_species, dim);

%#b = 0.1; 
b = 100.0;
c = 1.0;

% Discretize time interval
t = 0:dt:te; tlen = length(t);

% Discretize in space
[x, steps, nodes, A] = discretize_periodic(steps, square_len, Diff, Adv);

%# Both species treated separately!
%# Possible due to assumption of no coupling in diffusive term
% initial condition for u
u_old = 2*cos(sum(nodes, 2));
% initial condition for v
v_old = (b-c)*cos(sum(nodes,2));
u_old = {u_old, v_old};

[runtime, soln] = solve_ETD(dt, tlen, steps, A, u_old, @F);

u_soln = soln{1};
v_soln = soln{2};

Uex = (exp(-b-d)+exp(-c-d))*cos(sum(nodes, 2)-a);
Vex = (b-c)*exp(-c-d)*cos(sum(nodes, 2)-a);

Uex = reshape(Uex, steps, steps, steps);
u_ex = Uex;
Vex = reshape(Vex, steps, steps, steps);
Usoln = reshape(u_soln,steps,steps,steps); 
Vsoln = reshape(v_soln,steps,steps,steps);

disp(max(max(max(Usoln - Uex))));

if do_plot
plot_soln(Usoln, Vsoln, Uex, Vex, {x, x});

end
    

function Fr = F(u)
 f1 = -b*u{1} + u{2};
 f2 = -c*u{2};
 Fr = [f1 f2];
end

end

function plot_soln(Usoln, Vsoln, Uex, Vex, grid)
    
    x = grid{1};
    y = grid{2};

    Uplot = Usoln(:,:,end);
    Vplot = Vsoln(:,:,end);

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
    
    

  Uexplot = Uex(:,:,end);
  Vexplot = Vex(:,:,end);
  


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
