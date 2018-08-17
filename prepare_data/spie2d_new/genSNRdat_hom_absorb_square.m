clear all; close all; clc;

mcxseed=111;

N=100; % repeat N times with different rand seeds

%% set up the output dir

% Top-level Dir
topFolderName='./test_snr';
if ~exist(topFolderName, 'dir')  mkdir(topFolderName); end

%testDir = sprintf('%s/hom_square', topFolderName);
%if ~exist(testDir, 'dir')  mkdir(testDir); end

testDir = sprintf('%s/absorber_square', topFolderName);
if ~exist(testDir, 'dir')  mkdir(testDir); end

testDir_p5 = sprintf('%s/%1.0e', testDir, 1e5);
if ~exist(testDir_p5, 'dir')  mkdir(testDir_p5); end

testDir_p6 = sprintf('%s/%1.0e', testDir, 1e6);
if ~exist(testDir_p6, 'dir')  mkdir(testDir_p6); end

testDir_p7 = sprintf('%s/%1.0e', testDir, 1e7);
if ~exist(testDir_p7, 'dir')  mkdir(testDir_p7); end


%% generate N randseeds

rand_seed = randi([1 2^31-1], 1, N);
if (length(unique(rand_seed)) < length(rand_seed)) ~= 0
    error('There are repeated random seeds!')
end


%% read the image

% the test image is grayscle, text is black, bg is white
% size is 100 x 100
input_img = imread('./images/square.png');  % 
img_modify = uint8(input_img < 255); % make the text 1, others bg is 0 255
img_modify = img_modify + 1; % raise 1 to distinguish from the background
% imagesc(img_modify)



%%

for sid = 1:N
mcxseed = rand_seed(sid);

% 1e5
[cwdata, ~, ~] = rand_2d_mcx_grid_test(1e5, img_modify, 123, mcxseed);
currentImage = cwdata;
fname = sprintf('%s/test%d.mat', testDir_p5,  sid);
fprintf('Generating %s\n',fname);
feval('save', fname, 'currentImage');

% 1e6
[cwdata, ~, ~] = rand_2d_mcx_grid_test(1e6, img_modify, 123, mcxseed);
currentImage = cwdata;
fname = sprintf('%s/test%d.mat', testDir_p6,  sid);
fprintf('Generating %s\n',fname);
feval('save', fname, 'currentImage');

% 1e7
[cwdata, ~, ~] = rand_2d_mcx_grid_test(1e7, img_modify, 123, mcxseed);
currentImage = cwdata;
fname = sprintf('%s/test%d.mat', testDir_p7,  sid);
fprintf('Generating %s\n',fname);
feval('save', fname, 'currentImage');

end
