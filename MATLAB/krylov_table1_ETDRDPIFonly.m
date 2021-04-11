% Experiment of the Krylov Paper example 1

errors = zeros(4, 1);
times = zeros(4, 1);

a = randi([0,1], [100000, 2]);

% tic;
% 
% for i = 1:10000
%     aa = zeros([100000, 1]);
% end
% 
% toc

%for i = 0:3
for i = 0:6
    k =  0.005 / 2^(6-i);
    N = 10*(2^10);
%     k =  0.005 / 2^(3-i);
%     N = 10*(2^5);
    [time, u_old, Uex] = Periodic_benchmark_IFETDRDP_1D(k, N, false);
    
    times(i+1) = time;

    Usoln = reshape(u_old,N, 1);
    errors(i+1) = max(abs(Usoln-Uex));
    
%     Usoln = reshape(u_old,N, N);
%     errors(i+1) = max(max(abs(Usoln-Uex)));
    
%     Usoln = reshape(u_old,N,N,N);
%     errors(i+1) = max(max(max(abs(Usoln-Uex))));
    
    disp(["Time of iteration", i, ", k = ", k, ", N = ", N]);
    disp(times(i+1));
    disp(["Error of iteration", i]);
    disp(errors(i+1));
    
%     [time, u_old, Uex] = Periodic_benchmark_IFETDRDP(k, N, false);
%    
%     times(i+1) = time;
% 
%     Usoln = reshape(u_old,N,N,N);
%     errors(i+1) = max(max(max(abs(Usoln-Uex))));
%     
%     disp("Time");
%     disp(times(i+1));
%     disp("Error");
%     disp(errors(i+1)); 
    
    
end

for i = 0:5
   disp("Order");
   disp(log2(errors(i+2)/errors(i+1)));
end