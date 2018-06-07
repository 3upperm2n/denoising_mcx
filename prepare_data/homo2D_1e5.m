%% generating data sets : 2d homo
clc;
clear all;

addpath('./mcx')
addpath('./mcx/mcxlab')

%% reference
% [1] http://mcx.space/wiki/index.cgi?Doc/MCXLAB

%%

pho_cnt = 1e5;

%% Top-level Dir
topFolderName='../data/2d_homo';
if ~exist('../data/2d_homo/', 'dir')  mkdir(topFolderName); end

% e.g., mkdir  ./data/2d_homo/1e5/
dir_phn = sprintf('%s/%1.0e', topFolderName, pho_cnt);
if ~exist(dir_phn, 'dir')  mkdir(dir_phn); end

%%
x_samples = 20;
y_samples = 20;

x = 100;
y = 100;
z = 1;

volume = uint8(ones(z,y,x));

x_linspace = uint32(linspace(1,x,  x_samples));
y_linspace = uint32(linspace(1,y,  y_samples));
refractive_idx = [0.2, 0.4, 0.8, 1.0 , 1.2, 1.37, 1.4, 1.6, 1.8, 2.0];


%% Generate new random seed for Monte Carlo simulation

N = length(x_linspace) * length(y_linspace) * length(refractive_idx);

rand_seed = randi([1 2^31-1], 1, N);
if (length(unique(rand_seed)) < length(rand_seed)) ~= 0
    error('There are repeated random seeds!')
end




%%
testID = 1;
for i = 1:length(x_linspace)
   xpos = x_linspace(i);
   
   for j = 1:length(y_linspace)
      ypos = y_linspace(j);
      
      for k = 1:length(refractive_idx)
          refract = refractive_idx(k);
          %fprintf("%.2f %.2f %.2f %d\n", xpos, ypos, refract, testID);
          
          fname_save = sprintf('%s/test%d.mat', dir_phn, testID);
          %disp(fname);
          
          %
          % run simulation
          %
          homo2D_gen(pho_cnt, fname_save, rand_seed(testID), volume, xpos, ypos, refract)
          
          
          testID = testID + 1;
      end
   end
end
