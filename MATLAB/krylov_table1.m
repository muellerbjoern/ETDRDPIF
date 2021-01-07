% Experiment of the Krylov Paper example 1

errors = zeros(4, 1);
times = zeros(4, 1);

for i = 0:3
    k = 0.005 / (2^i);
    N = 10*(2^i);
    [time, u_old, Uex] = Periodic_benchmark_IFETDRDP(k, N, false);
    
    times(i+1) = time;

    Usoln = reshape(u_old,N,N,N);
    errors(i+1) = max(max(max(abs(Usoln-Uex))));
    
    disp("Time");
    disp(times(i+1));
    disp("Error");
    disp(errors(i+1));
end