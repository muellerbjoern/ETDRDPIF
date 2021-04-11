i = 1;
for te = [0.025, 0.125, 0.15, 0.2, 0.225, 5]
   Schnakenberg_2D_IFETDRDP(te, 0.000125, 20, true);
   saveas(i, sprintf('~/Documents/Masterarbeit/Arbeit/figures/bhatt_locally_fig13%c.eps', char(i+96)), 'epsc');
   view([0, 0, 1]);
   saveas(i, sprintf('~/Documents/Masterarbeit/Arbeit/figures/bhatt_locally_fig14%c.eps', char(i+96)), 'epsc');
   disp(char(i+96));
   i = i+1;
end