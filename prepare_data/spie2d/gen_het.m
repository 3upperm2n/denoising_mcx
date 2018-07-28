clc; clear all;

%%

% Top-level Dir
topFolderName='../../data/spie2d';
if ~exist(topFolderName, 'dir')  mkdir(topFolderName); end

testDir = sprintf('%s/het', topFolderName);
if ~exist(testDir, 'dir')  mkdir(testDir); end

testDir_p4 = sprintf('%s/%1.0e', testDir, 1e4);
if ~exist(testDir_p4, 'dir')  mkdir(testDir_p4); end

testDir_p7 = sprintf('%s/%1.0e', testDir, 1e7);
if ~exist(testDir_p7, 'dir')  mkdir(testDir_p7); end



%%
rng(123456789,'twister');


imSize = [100 100];
maxObj = 10;      % generate 1 to 10 objs per image
maxRepeat = 2;    % repeat the same simulation 
N = 100;          % randomize text objs 

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
    % 1e4
    %
    [cwdata, imdata]=gentraining(1e4, imSize, maxObj, maxRepeat, mySeed);

    % export the image
    for ib = 1:maxObj
       for ir = 1:maxRepeat
           currentImage = cwdata(:,:,ir,ib);
           
           local_id = ir + (ib - 1) * maxRepeat;
           testID = offset + local_id;
%            fprintf('[%s] \t testID = %d\n', phn, testID);
           
           fname = sprintf('%s/test%d.mat', testDir_p4,  testID);
           fprintf('Generating %s\n',fname);
           feval('save', fname, 'currentImage');

       end
    end
    
 
    %
    % 1e7
    %
    [cwdata_, imdata_]=gentraining(1e7, imSize, maxObj, maxRepeat, mySeed); % modify config here

    % export the image
    for ib = 1:maxObj
       for ir = 1:maxRepeat
           currentImage = cwdata_(:,:,ir,ib);
           
           local_id = ir + (ib - 1) * maxRepeat;
           testID = offset + local_id;
           
           fname = sprintf('%s/test%d.mat', testDir_p7,  testID); % modify name here
           fprintf('Generating %s\n',fname);
           feval('save', fname, 'currentImage');

       end
    end
   
%     break
end

