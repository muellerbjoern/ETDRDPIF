% Experiment of the Krylov Paper example 1

errors = zeros(4, 1);
times = zeros(4, 1);
N = 32;
u_curr = zeros(N*N*N);
u_curr_k = zeros(N*N*N);
for i = 1:5
    k = 0.01 / (2^i);
    u_old = u_curr;
    u_old_k = u_curr_k;
    [time, u_curr] = Schnakenberg_3D_IFETDRDP_periodic(1, k, N, false);
    [time_k, u_curr_k] = Krylov_Schnakenberg_periodic(1, k, N);
    %u_curr = reshape(u_curr,N,N,N);
   
    % Starting from second iteration, as the error is always the error of
    % the previous iteration
    if i > 1
    errors(i-1) = norm(u_curr - u_old, inf);
    error_k = norm(u_curr_k - u_old_k, inf);
   
    disp(['Error of iteration ', num2str(i-1)]);
    disp(['    ETD-RDP-IF: ', num2str(errors(i-1))]);
    disp(['    Krylov-ETD: ',num2str(error_k)]);
    disp(' ');
    disp(' ');
    end
    

    times(i) = time;
    disp(['Time of iteration ', num2str(i), ', k = ', num2str(k), ', N = ', num2str(N)]);
    disp(['    ETD-RDP-IF: ', num2str(times(i))]);
    disp(['    Krylov-ETD: ', num2str(time_k)]);
    disp(' ');
    disp("maximal Difference ETD-RDP-IF - Krylov-ETD");
    disp(norm(u_curr - u_curr_k, inf));
    
end