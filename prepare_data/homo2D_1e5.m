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
          %fprintf('%d %d %.2f %d\n', xpos, ypos, refract, testID);
          
          fname_save = sprintf('%s/test%d.mat', dir_phn, testID);
          %disp(fname);
          
          %--------------------
          %  start simulation
          %--------------------
          %homo2D_gen(pho_cnt, fname_save, rand_seed(testID), volume, xpos, ypos, refract)
          
          clear cfg
          cfg.nphoton=pho_cnt;
          cfg.vol= volume;
          
          %cfg.srcpos=[1 1 1];  % change the last two pos [z, y, x]
          %srcpos_str = strcat("cfg.srcpos=[1 ", string(ypos), " ", string(xpos), "];");
          srcpos_str = sprintf("cfg.srcpos=[1 %d %d];", ypos, xpos);
          
          %disp(srcpos_str);
          %eval('cfg.srcpos=[1 1 1];')
          eval(srcpos_str);
          
          cfg.srcdir=[0 0 1];
          cfg.gpuid=1;
          cfg.autopilot=1;
          %
          % configure optical properties here 
          %
          % cfg.prop=[0 0 1 1;0.005 1 0 1.37]; % scattering mus, anisotropy, mua, refractive index
          cfg.prop=[0 0 1 1;0.005 1 0 refract];
          cfg.tstart=0;
          cfg.tend=5e-8;
          cfg.tstep=5e-8;
          cfg.seed = rand_seed(testID); % each random seed will have different pattern 

          % calculate the flux distribution with the given config
          [flux,~]=mcxlab(cfg);

          currentImage = squeeze(sum(flux.data,4));
          feval('save', fname_save, 'currentImage');
          
          testID = testID + 1;
          
          %break
      end
      %break
   end
   %break
end
