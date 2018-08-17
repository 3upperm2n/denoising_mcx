clear all; close all; clc;

%% Top-level Dir
topFolderName='../../data/spie2d_new';
if ~exist(topFolderName, 'dir')  mkdir(topFolderName); end

testDir = sprintf('%s/randsq', topFolderName);
if ~exist(testDir, 'dir')  mkdir(testDir); end

testDir_p5 = sprintf('%s/%1.0e', testDir, 1e5);
if ~exist(testDir_p5, 'dir')  mkdir(testDir_p5); end

testDir_p7 = sprintf('%s/%1.0e', testDir, 1e7);
if ~exist(testDir_p7, 'dir')  mkdir(testDir_p7); end



%%

rng(123456789,'twister');

large_num = 9992039;


%%

% generate obj locations in image 100x100
%
obj_r = [30, 50, 70];
obj_c = [30, 50, 70];

rows = length(obj_r);
cols = length(obj_c);

obj_locs = zeros(rows * cols,2);
rowID = uint32(1);
for ii = 1 : rows
    for jj = 1 : cols
        obj_locs(rowID,:)= [obj_r(ii) obj_c(jj)];
        rowID = rowID + uint32(1);
    end
end


%%

% for each loc, generate 1 square with a rand shape and rand prop
% we iterate 100 times for all the locs

N = 200;
totalPos = rows * cols;

% generate N randseeds
rand_seed = randi([1 2^31-1], 1, N * totalPos);
if (length(unique(rand_seed)) < length(rand_seed)) ~= 0
    error('There are repeated random seeds!')
end


% each iteration
sid = uint32(1);
for ii = 1: N
    fprintf("\n");
    % go through each location 
    
    for jj = 1: totalPos
        % image bg
        img = ones(100,100);
        
        % obj loc
        c_rl = obj_locs(jj, :);
        c_r = c_rl(1);
        c_l = c_rl(2);
        %fprintf("%d, %d\n", c_r, c_l);
        
        h = 5+randi(10); % 6 to 15
        img(c_r-h:c_r+h, c_l-h:c_l+h)=2; % obj shape/size
        img = uint8(img);
        fprintf("c_r: %d \t c_l:%d \t h: %d\n", c_r, c_l, h);
        
        % run mcx simulation
        % mcxseed should be a large nummber (here we fix it for demo)
        rseed = rand_seed(sid);
        
        [cwdata, ~, ~]  = rand_2d_mcx_grid_test(1e5, img, rseed, large_num + rseed);
        [cwdata1, ~, ~] = rand_2d_mcx_grid_test(1e7, img, rseed, large_num + rseed);
        
        
        % export the image
        currentImage = cwdata(:,:,1,1);
        fname = sprintf('%s/test%d.mat', testDir_p5,  sid);
        fprintf('Generating %s\n',fname);
        feval('save', fname, 'currentImage');
        
        currentImage = cwdata1(:,:,1,1);
        fname = sprintf('%s/test%d.mat', testDir_p7,  sid);
        fprintf('Generating %s\n',fname);
        feval('save', fname, 'currentImage');


        
        sid = sid + uint32(1);
        
        %break;
    end
    
    %break;
end