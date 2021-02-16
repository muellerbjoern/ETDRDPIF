function [runtime,u_soln,u_ex] = Schnakenberg_2D_IFETDRDP(dt,steps,do_plot)

% dt: time step. Default is 0.001
% steps: number of spatial points in each coordinate direction. Default is 11

%# k is temporal discretization (dt);
%# h is spatial discretization (steps);
steps = 20;
dt = 0.000125;
dim = 2;
num_species = 2;

te = 5;
square_len = 1.0;

% Discretize time interval
t = 0:dt:te; tlen = length(t);

% Discretize in space
[x, nodes, B, C] = discretize_Neumann(steps, square_len, dim);

% Number of points may be different from initial value of steps
% Steps determines number of sub-intervals that the interval in
% each dimension is split into!
% Dirichlet allows removing both end points, Neumann requires both
steps = size(B, 1);


%% Model Paramters and initial conditions
a1 = 0;
a2 = 0;
Adv = zeros(num_species, dim);
Adv(1, :) = a1;
Adv(2, :) = a2;

d1 = 1.0;
d2 = 10.0;
Diff = zeros(num_species, dim);
Diff(1, :) = d1;
Diff(2, :) = d2;

a = 0.126779;
b = 0.792366;
gamma = 1000;

cos_sum = zeros(size(nodes(:, 1)));
for j = 1:8
    cos_sum = cos_sum + cos(2*pi*j*nodes(:, 1));
end

u_old = 0.919145 + 0.0016*cos(2*pi*(nodes(:, 1) + nodes(:, 2))) + 0.01*cos_sum;
v_old = 0.937903 + 0.0016*cos(2*pi*(nodes(:, 1) + nodes(:, 2))) + 0.01*cos_sum;
u_old = {u_old, v_old};

[runtime, soln] = solve_ETD(dt, tlen, B, C, Diff, Adv, u_old, @F);

u_soln = soln{1};
v_soln = soln{2};

% Uex = (exp(-b-d)+exp(-c-d))*cos(sum(nodes, 2)-a);
% Vex = (b-c)*exp(-c-d)*cos(sum(nodes, 2)-a);
% 
% Uex = reshape(Uex, steps, steps, steps);
% Vex = reshape(Vex, steps, steps, steps);

Usoln = reshape(u_soln,steps,steps); 
Vsoln = reshape(v_soln,steps,steps);

% disp(max(max(max(Usoln - Uex))));

if do_plot
plot_soln(Usoln, Vsoln, {x, x});

end
    

function Fr = F(u)
 f1 = (-u{1} + u{1}.^2.*u{2} + a)*gamma;
 f2 = (-u{1}.^2.*u{2} + b)*gamma;
 Fr = [f1 f2];
end

end




function plot_soln(Usoln, Vsoln, grid)
    
    x = grid{1};
    y = grid{2};

    Uplot = Usoln(:,:);
    Vplot = Vsoln(:,:);
    
    figure(49)
    % Transpose Usoln! See contourf documentation
    surf(x,y, Uplot', 'FaceColor', 'interp')
    xlabel('x')
    ylabel('y')
    title("U")
    zlim([0, 2]);
    colormap(jet);
    colorbar
    set(gca,'LineWidth', 1);
    set(gca,'FontSize',10);
    set(gca,'FontWeight','bold');
    pbaspect(gca,[1 1 1])
    %#set(gca,'XTick',[0 0.2 0.4 0.6 0.8 1]);
    %#set(gca,'YTick',[0 0.2 0.4 0.6 0.8 1]);
    %#print -depsc2 sliceu.eps
    
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
