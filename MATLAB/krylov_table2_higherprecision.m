% Experiment of the Krylov Paper example 1

errors = zeros(4, 1);
times = zeros(4, 1);
N = 32;
u_curr = zeros(N*N*N);
u_curr_k = zeros(N*N*N);

% k = 0.01/256;
% [~, u_comp2] = Brusselator_ADR(1, k, N, false);
% [~, u_comp_k2]  = Krylov_Brusselator3DRDA(1, k, N);
% 
% save('u_comp2.mat', 'u_comp2');
% save('u_comp_k2.mat', 'u_comp_k2');
% 
% k = 0.01/512;
% [~, u_comp3] = Brusselator_ADR(1, k, N, false);
% [~, u_comp_k3]  = Krylov_Brusselator3DRDA(1, k, N);
% 
% save('u_comp3.mat', 'u_comp3');
% save('u_comp_k3.mat', 'u_comp_k3');

% k = 0.01/1024;
% [~, u_comp4] = Brusselator_ADR(1, k, N, false);
% save('u_comp4.mat', 'u_comp4');
% [~, u_comp_k4]  = Krylov_Brusselator3DRDA(1, k, N);
% save('u_comp_k4.mat', 'u_comp_k4');

u_comp = load('u_comp.mat').u_comp;
u_comp_k = load('u_comp_k.mat').u_comp_k;
u_comp2 = load('u_comp2.mat').u_comp2;
u_comp_k2 = load('u_comp_k2.mat').u_comp_k2;
u_comp3 = load('u_comp3.mat').u_comp3;
u_comp_k3 = load('u_comp_k3.mat').u_comp_k3;
u_comp4 = load('u_comp4.mat').u_comp4;
u_comp_k4 = load('u_comp_k4.mat').u_comp_k4;

disp(norm(u_comp - u_comp2, 'inf'));
disp(norm(u_comp - u_comp3, 'inf'));
disp(norm(u_comp2 - u_comp3, 'inf'));
disp(norm(u_comp3 - u_comp4, 'inf'));
disp(norm(u_comp_k2 - u_comp2, 'inf'));
disp(norm(u_comp_k3 - u_comp3, 'inf'));
disp(norm(u_comp_k4 - u_comp4, 'inf'));
disp(norm(u_comp_k3 - u_comp_k4, 'inf'));

u_comp = u_comp3;
u_comp_k = u_comp_k3;

return;

for i = 1:5
    k = 0.01 / (2^i);
    u_old = u_curr;
    u_old_k = u_curr_k;
    [time, u_curr] = Brusselator_ADR(1, k, N, false);
    [time_k, u_curr_k] = Krylov_Brusselator3DRDA(1, k, N);
    %u_curr = reshape(u_curr,N,N,N);
   
    % Starting from second iteration, as the error is always the error of
    % the previous iteration
    %if i > 1
    %end
    

    times(i) = time;
    disp(['Time of iteration ', num2str(i), ', k = ', num2str(k), ', N = ', num2str(N)]);
    disp(['    ETD-RDP-IF: ', num2str(times(i))]);
    disp(['    Krylov-ETD: ', num2str(time_k)]);
    errors(i) = norm(u_curr - u_comp, inf);
    error_k = norm(u_curr_k - u_comp_k, inf);
    disp(['Error of iteration ', num2str(i)]);
    disp(['    ETD-RDP-IF: ', num2str(errors(i))]);
    disp(['    Krylov-ETD: ',num2str(error_k)]);
    disp(['    ETD-RDP-IF comp Krylov: ', num2str(norm(u_curr - u_comp_k, inf))]);
    disp(['    Krylov-ETD comp RDP-IF: ',num2str(norm(u_curr_k - u_comp, inf))]);
    disp(' ');
    disp(' ');
    disp(' ');
    disp("maximal Difference ETD-RDP-IF - Krylov-ETD");
    disp(norm(u_curr - u_curr_k, inf));
    
end