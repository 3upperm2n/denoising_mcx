clc; clear all;

%%

% Top-level Dir
topFolderName='../../data/spie2d';
if ~exist(topFolderName, 'dir')  mkdir(topFolderName); end

testDir = sprintf('%s/hom', topFolderName);
if ~exist(testDir, 'dir')  mkdir(testDir); end

testDir_p4 = sprintf('%s/%1.0e', testDir, 1e4);
if ~exist(testDir_p4, 'dir')  mkdir(testDir_p4); end

testDir_p7 = sprintf('%s/%1.0e', testDir, 1e7);
if ~exist(testDir_p7, 'dir')  mkdir(testDir_p7); end



%%
rng(123456789,'twister');


imSize = [100 100];
N = 2000;          % randomize text (with location) objs 

% generate N randseeds
rand_seed = randi([1 2^31-1], 1, N);
if (length(unique(rand_seed)) < length(rand_seed)) ~= 0
    error('There are repeated random seeds!')
end


for sid = 1:N
    mySeed = rand_seed(sid);
    % fprintf('seed %d\n', mySeed);

    
    % cwdata: simulation image
    % imdata: text image
    % cwdata: 4D  = (x,y,repeat, objs)
    
    %
    % 1e4
    %
    [cwdata, imdata]=gentraining_grid(1e4, imSize, 0, 1, mySeed);

    % export the image
	currentImage = cwdata(:,:,1,1);
    fname = sprintf('%s/test%d.mat', testDir_p4,  sid);
    fprintf('Generating %s\n',fname);
    feval('save', fname, 'currentImage');

    %
    % 1e7
    %
    [cwdata_, imdata_]=gentraining(1e7, imSize, 0, 1, mySeed); % modify config here

    currentImage = cwdata_(:,:,1,1);
    fname = sprintf('%s/test%d.mat', testDir_p7,  sid); % modify name here
    fprintf('Generating %s\n',fname);
    feval('save', fname, 'currentImage');

   
end

