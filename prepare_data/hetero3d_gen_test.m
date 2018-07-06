%% generating data sets used in OSA

clc;
close all;
clear all;

addpath('./mcx')
addpath('./mcx/mcxlab')


% preparing the input data
% structure for sphere-diffusion toolbox for the analytical solution

clear ana

ana.v=299792458000;  % mm/s
ana.a=10;            % radius, mm
ana.omua=0.02;      % outside mua 1/mm  : default 0.002
ana.omusp=1.0;       % outside mus' 1/mm
ana.imua=0.05; % org 0.05
ana.imusp=0.05;
ana.src=[30,30,0];
ana.maxl=20;
ana.omega=0;

% set seed to make the simulation repeatible
cfg.seed=hex2dec('623F9A9E'); 

% cfg.nphoton=1e5;
cfg.nphoton=1e8;

% define a 1cm radius sphere within a 6x6x6 cm box
dim=100;
[xi,yi,zi]=meshgrid(1:dim,1:dim,1:dim);
dist=(xi-30.5).^2+(yi-30.5).^2+(zi-30.5).^2;
cfg.vol=ones(size(xi));
cfg.vol(dist<ana.a*ana.a)=5;  % increase from 2 to 5
cfg.vol=uint8(cfg.vol);

% define the source position
cfg.srcpos=[30,30,0]+1;
cfg.srcdir=[0 0 1];

% format: [mua(1/mm) mus(1/mm) g n]
cfg.prop=[0 0 1 1          % medium 0: the environment
   ana.omua ana.omusp/(1-0.01) 0.01 1.37     % medium 1: cube
   ana.imua ana.imusp/(1-0.01) 0.01 1.37];   % medium 2: spherical inclusion

% time-domain simulation parameters
cfg.tstart=0;
cfg.tend=5e-8;
cfg.tstep=5e-8;

% GPU thread configuration
cfg.autopilot=1;
cfg.gpuid=1;

cfg.isreflect=0; % disable reflection at exterior boundaries

% calculate the flux distribution with the given config
[flux,~]=mcxlab(cfg);

image3D = flux.data;

size(image3D)

image_sel = image3D(:,31,:);
currentImage = squeeze(image_sel);

imagesc(log10(abs(currentImage)), [-3 7]);

%%


%
% save  the images to mat file
%

% Top-level Dir                                                                 
topFolderName='../data/hetero3d';                                                    
if ~exist('../data/hetero3d/', 'dir')  mkdir(topFolderName); end

dir_phn = sprintf('%s/%1.0e', topFolderName, cfg.nphoton);                 
if ~exist(dir_phn, 'dir')  mkdir(dir_phn); end 

for imgID  = 1 : 100
fname = sprintf('%s/img_%d.mat', dir_phn, imgID);
fprintf('Generating %s (y-axis)\n',fname);                          
currentImage = squeeze(image3D(:,imgID,:));                       
feval('save', fname, 'currentImage');
end