% Experiment of the Krylov Paper example 1

errors = zeros(4, 1);
times = zeros(4, 1);
for i = 0:3
    k = 0.005 / (2^i);
    N = 10*(2^i);
    [time, u_curr, u_ex] = Periodic_benchmark_IFETDRDP_refactored(k, N, false);
    [time_k, u_curr_k] = Krylov_Periodic_benchmark(1.0, k, N);
    %u_curr = reshape(u_curr,N,N,N);
    
    if i > -1
    times(i+1) = time;
    u_ex = reshape(u_ex, N*N*N, 1);
    disp(size(u_ex));
    errors(i+1) = norm(u_curr - u_ex, inf);
    error_k = norm(u_curr_k - u_ex, inf);
    
    disp("Time");
    disp(times(i+1));
    disp(time_k);
    disp("Error");
    disp(errors(i+1));
    disp(error_k)
    
    disp("Difference");
    disp(norm(u_curr - u_curr_k, inf));
    
    end
    
end