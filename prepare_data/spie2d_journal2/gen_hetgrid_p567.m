clc; clear all;

%%

% Top-level Dir
%topFolderName='../../data/spie2d';

topFolderName='../../data/spie_2d_hetgrid_p567';
if ~exist(topFolderName, 'dir')  mkdir(topFolderName); end

testDir = topFolderName;
%testDir = sprintf('%s/het_grid', topFolderName);
%if ~exist(testDir, 'dir')  mkdir(testDir); end

testDir_p5 = sprintf('%s/%1.0e', testDir, 1e5);
if ~exist(testDir_p5, 'dir')  mkdir(testDir_p5); end

testDir_p6 = sprintf('%s/%1.0e', testDir, 1e6);
if ~exist(testDir_p6, 'dir')  mkdir(testDir_p6); end

testDir_p7 = sprintf('%s/%1.0e', testDir, 1e7);
if ~exist(testDir_p7, 'dir')  mkdir(testDir_p7); end



%%
rng(123456789,'twister');


imSize = [100 100];
maxObj = 10;      % generate 1 to 10 objs per image
maxRepeat = 2;    % repeat the same simulation 
N = 500;          % randomize text objs 

% generate N randseeds
rand_seed = randi([1 2^31-1], 1, N);
if (length(unique(rand_seed)) < length(rand_seed)) ~= 0
    error('There are repeated random seeds!')
end

stepSize = maxObj * maxRepeat;

for sid = 1:N
    mySeed = rand_seed(sid);
    % fprintf('seed %d\n', mySeed);

    offset = stepSize * (sid -1);
    
    % cwdata: simulation image
    % imdata: text image
    % cwdata: 4D  = (x,y,repeat, objs)
    
    %
    % 1e5
    %
    %[cwdata, imdata]=gentraining_grid(1e4, imSize, maxObj, maxRepeat, mySeed);
    [cwdata, ~]    =gentraining_grid(1e5, imSize, maxObj, maxRepeat, mySeed);
    [cwdata_p6, ~] =gentraining_grid(1e6, imSize, maxObj, maxRepeat, mySeed); % modify config here
    [cwdata_p7, ~] =gentraining_grid(1e7, imSize, maxObj, maxRepeat, mySeed); % modify config here

    
    %break;
    
    % export the image
    for ib = 1:maxObj
       for ir = 1:maxRepeat
           local_id = ir + (ib - 1) * maxRepeat;
           testID = offset + local_id;

           currentImage = cwdata(:,:,ir,ib);
		   % fprintf('[%s] \t testID = %d\n', phn, testID);
           fname = sprintf('%s/test%d.mat', testDir_p5,  testID);
           fprintf('Generating %s\n',fname);
           feval('save', fname, 'currentImage');


           currentImage = cwdata_p6(:,:,ir,ib);
           fname = sprintf('%s/test%d.mat', testDir_p6,  testID); % modify name here
           fprintf('Generating %s\n',fname);
           feval('save', fname, 'currentImage');

           currentImage = cwdata_p7(:,:,ir,ib);
           fname = sprintf('%s/test%d.mat', testDir_p7,  testID); % modify name here
           fprintf('Generating %s\n',fname);
           feval('save', fname, 'currentImage');
        end
    end
    
   
%     break
end

