% simple test

clc;
close all;
clear all;

addpath('../mcx')
addpath('../mcx/mcxlab')

%  [cw, myimg, cfg]=rand_2d_mcx(nphoton, maxprop, imsize, randseed, srcoffset)

% [cw, myimg, ~] = rand_2d_mcx(1e4, 1, [100 100], 1234, [0 0]);
% plot_rand_mcx(cw, myimg)
%  
% [cw, myimg, ~] = rand_2d_mcx(1e4, 1, [100 100], 1234, [0 10]);
% plot_rand_mcx(cw, myimg)
% 
% % [cw, myimg, ~] = rand_2d_mcx(1e4, 1, [100 100], 1234, [0 50]);
% % plot_rand_mcx(cw, myimg)
% 
% 
% % [cw, myimg, ~] = rand_2d_mcx(1e4, 1, [100 100], 1234, [0 99]);
% % plot_rand_mcx(cw, myimg)
% 
% [cw, myimg, ~] = rand_2d_mcx(1e4, 1, [100 100], 1234, [49 49]);
% plot_rand_mcx(cw, myimg)
% 
% [cw, myimg, ~] = rand_2d_mcx(1e4, 1, [100 100], 1234, [20 20]);
% plot_rand_mcx(cw, myimg)

[cw, myimg, ~] = rand_2d_mcx(1e4, 0, [100 100], 123);
plot_rand_mcx(cw, myimg)

clear cw; clear myimg;

[cw, myimg, ~] = rand_2d_mcx(1e6, 0, [100 100], 123);
plot_rand_mcx(cw, myimg)

clear cw; clear myimg;

[cw, myimg, ~] = rand_2d_mcx(1e4, 7, [100 100], 123);
plot_rand_mcx(cw, myimg)

clear cw; clear myimg;

[cw, myimg, ~] = rand_2d_mcx(1e6, 7, [100 100], 123);
plot_rand_mcx(cw, myimg)
