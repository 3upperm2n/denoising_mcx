%% generating data sets used in OSA
clc;
clear all;

addpath('./mcx')
addpath('./mcx/mcxlab')

%% reference
% [1]http://mcx.space/wiki/index.cgi?Doc/MCXLAB

%% configuration
% 20 evenly separated loc on x-axis, 10 evenly separated loc on y-axis
% total 20 x 10 = 200 tests

%% Top-level Dir
topFolderName='../data/srcpos_xy';
if ~exist('../data/srcpos_xy/', 'dir')  mkdir(topFolderName); end


%%
x_samples = 20;
y_samples = 10;
N = x_samples * y_samples;   % total tests for each simulation volume

x = 100;
y = 100;
z = 100;

volume = uint8(ones(x,y,z));

x_linspace = uint32(linspace(0,x,  x_samples));
y_linspace = uint32(linspace(0,y/2,y_samples));

srcloc_2d = zeros(N,2);
for i=1:N
    row_id = floor((i + x_samples - 1) / x_samples);
    col_id = mod(i-1,x_samples) + 1;
    srcloc_2d(i,:) = [y_linspace(row_id) x_linspace(col_id)];
end

%%

pho_cnt = [1e5, 1e8];

[~, sim_phn] = size(pho_cnt);

for k=1:sim_phn
    
    %
    % Generate new random seed for Monte Carlo simulation
    %
    rand_seed = randi([1 2^31-1], 1, N);
    if (length(unique(rand_seed)) < length(rand_seed)) ~= 0
        error('There are repeated random seeds!')
    end

    % e.g., mkdir  ./srcpos_xy/1e5/
    dir_phn = sprintf('./%s/%1.0e', topFolderName, pho_cnt(k));
    if ~exist(dir_phn, 'dir')  mkdir(dir_phn); end
    
    
    for tid = 1:N
        % e.g., mkdir  ./srcpos_xy/1e5/1
        dir_phn_test = sprintf('%s/%d', dir_phn, tid);
        if ~exist(dir_phn_test, 'dir')  mkdir(dir_phn_test); end
        
        % select the x,y src loc 
        y_loc = srcloc_2d(tid,1);
        x_loc = srcloc_2d(tid,2);
        
        if y_loc == 0
            y_loc = 1; 
        end
        
        if x_loc == 0
            x_loc = 1; 
        end
        
        gen_images(tid, rand_seed(tid), pho_cnt(k), volume, ...
        x_loc, y_loc,x,y,z,...
        dir_phn_test)
    
        %break
    end
    
    %break
end


%%
function gen_images(tid, rand_seed, pho_cnt, volume, x_loc, y_loc,x,y,z, ...
    dir_phn_test)

    clear cfg
    cfg.nphoton=pho_cnt;
    
    cfg.vol= volume;
    cfg.srcpos=[x_loc y_loc 1];
    cfg.srcdir=[0 0 1]; % perpendicular to y
    cfg.gpuid=1;
    cfg.autopilot=1;
    
    %
    % configure optical properties here 
    %
    cfg.prop=[0 0 1 1;0.005 1 0 1.37];
    cfg.tstart=0;
    cfg.tend=5e-8;
    cfg.tstep=5e-8;
    cfg.seed = rand_seed; % each random seed will have different pattern 

    % calculate the flux distribution with the given config
    %[flux,detpos]=mcxlab(cfg);
    [flux,~]=mcxlab(cfg);

    image3D=flux.data;


    %------
    % x-axis
    %------
    dir_x = sprintf('%s/x', dir_phn_test);
    if ~exist(dir_x, 'dir')  mkdir(dir_x); end
        
    for imageID=1:x
        fname = sprintf('%s/osa_phn%1.0e_test%d_img%d.mat', dir_x, pho_cnt, tid, imageID);
        fprintf('Generating %s\n (x-axis)',fname);
        currentImage = squeeze(image3D(imageID,:,:));
        feval('save', fname, 'currentImage');
    end

    
    %------
    % y-axis
    %------
    dir_y = sprintf('%s/y', dir_phn_test);
    if ~exist(dir_y, 'dir')  mkdir(dir_y); end

    for imageID=1:y
        fname = sprintf('%s/osa_phn%1.0e_test%d_img%d.mat', dir_y, pho_cnt, tid, imageID);
        fprintf('Generating %s\n (y-axis)',fname);
        currentImage = squeeze(image3D(:,imageID,:));
        feval('save', fname, 'currentImage');
    end


    %------
    % z-axis
    %------
    dir_z = sprintf('%s/z', dir_phn_test);
    if ~exist(dir_z, 'dir')  mkdir(dir_z); end

    for imageID=1:z
        fname = sprintf('%s/osa_phn%1.0e_test%d_img%d.mat', dir_z, pho_cnt, tid, imageID);
        fprintf('Generating %s\n (z-axis)',fname);
        currentImage = squeeze(image3D(:,:,imageID));
        feval('save', fname, 'currentImage');
    end

end