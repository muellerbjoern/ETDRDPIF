% Experiment of the Krylov Paper example 1

errors = zeros(4, 1);
times = zeros(4, 1);

a = randi([0,1], [100000, 2]);

tic;

for i = 1:10000
    aa = zeros([100000, 1]);
end

toc

%for i = 0:3
for i = 1:3
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
    
    [time, u_old, Uex] = Periodic_benchmark_IFETDRDP_FFT2(k, N, false);
    
    times(i+1) = time;

    Usoln = reshape(u_old,N,N,N);
    errors(i+1) = max(max(max(abs(Usoln-Uex))));
    
    disp("Time");
    disp(times(i+1));
    disp("Error");
    disp(errors(i+1)); 
    
    
end