% Experiment of the Krylov Paper example 1

errors = zeros(4, 1);
times = zeros(4, 1);
for i = 0:0
    k = 0.005 / (2^i);
    N = 10*(2^i);
    [time, u_curr, u_ex] = Periodic_benchmark_IFETDRDP_refactored(k, N, false);
    [time_k, u_curr_k] = Krylov_Periodic_benchmark(1.0, k, N);
    %u_curr = reshape(u_curr,N,N,N);
    
    if i > -1
    times(i) = time;
    errors(i) = norm(u_curr - u_ex, inf);
    error_k = norm(u_curr_k - u_ex, inf);
    
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