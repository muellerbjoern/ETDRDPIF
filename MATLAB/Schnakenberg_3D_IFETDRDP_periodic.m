function [runtime,u_soln,u_ex] = Schnakenberg_3D_IFETDRDP_periodic(te, dt,steps,do_plot)

% dt: time step. Default is 0.001
% steps: number of spatial points in each coordinate direction. Default is 11

%# k is temporal discretization (dt);
%# h is spatial discretization (steps);
dim = 3;
num_species = 2;

square_len = 1.0;


%% Model Paramters and initial conditions
a1 = 1.0;
a2 = 1.0;
Adv = zeros(num_species, dim);
Adv(1, :) = -a1;
Adv(2, :) = -a2;

d1 = .05;
d2 = .01;
Diff = zeros(num_species, dim);
Diff(1, :) = d1;
Diff(2, :) = d2;

a = 1.0;
b = 0.9;
gamma = 1.0;
Lambda = square_len;

% Discretize time interval
t = 0:dt:te; tlen = length(t);

% Discretize in space
% Number of points may be different from initial value of steps
% Steps determines number of sub-intervals that the interval in
% each dimension is split into!
% Dirichlet allows removing both end points, Neumann requires both
[x, steps, nodes, A] = discretize_periodic(steps, square_len, Diff, Adv);

% Commutativity check; was true
% for ref = 1:num_species
%     for ref2 = 1:dim
%         for ref3 = 1:dim
%             isequal(A{ref, ref2} * A{ref, ref3}, A{ref, ref3} * A{ref, ref2})
%         end
%     end
% end


%# Both species treated separately!
%# Possible due to assumption of no coupling in diffusive term
% initial condition for u
u_old = 1.0 - exp(-10*((nodes(:, 1)-Lambda/2).^2 + (nodes(:, 2)-Lambda/2).^2 + (nodes(:, 3)-Lambda/2).^2));
% initial condition for v
v_old = u_old - 0.1;
u_old = {u_old, v_old};

[runtime, soln] = solve_ETD(dt, tlen, steps, A, u_old, @F);

u_soln = soln{1};
v_soln = soln{2};

% Uex = (exp(-b-d)+exp(-c-d))*cos(sum(nodes, 2)-a);
% Vex = (b-c)*exp(-c-d)*cos(sum(nodes, 2)-a);
% 
% Uex = reshape(Uex, steps, steps, steps);
% Vex = reshape(Vex, steps, steps, steps);

Usoln = reshape(u_soln,steps,steps,steps); 
Vsoln = reshape(v_soln,steps,steps,steps);

% disp(max(max(max(Usoln - Uex))));

% if do_plot
% plot_soln(Usoln, Vsoln, {x, x}, te);
% 
% end
    

function Fr = F(u)
 f1 = (-u{1} + u{1}.^2.*u{2} + a)*gamma;
 f2 = (-u{1}.^2.*u{2} + b)*gamma;
 Fr = [f1 f2];
end

end




function plot_soln(Usoln, Vsoln, grid, te)
    
    x = grid{1};
    y = grid{2};

    Uplot = Usoln(:,:,1);
    Vplot = Vsoln(:,:,end);
    
    for i_plot = [15 16] %[5 8 9 15 16 17]
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
