% Experiment of the Krylov Paper example 1

errors = zeros(4, 1);
times = zeros(4, 1);
N = 32;
u_curr = zeros(N*N*N);
u_curr_k = zeros(N*N*N);
for i = 0:4
    k = 0.01 / (2^i);
    u_old = u_curr;
    u_old_k = u_curr_k;
    [time, u_curr] = Brusselator_ADR(1.0, k, N, false);
    [time_k, u_curr_k] = Krylov_Brusselator3DRDA(1.0, k, N);
    %u_curr = reshape(u_curr,N,N,N);
    
    if i > 0
    times(i) = time;
    errors(i) = norm(u_curr - u_old, inf);
    error_k = norm(u_curr_k - u_old_k, inf);
    
    disp("Time");
    disp(times(i));
    disp(time_k);
    disp("Error");
    disp(errors(i));
    disp(error_k)
    
    disp("Difference");
    disp(norm(u_curr - u_curr_k, inf));
    
    end
    
end