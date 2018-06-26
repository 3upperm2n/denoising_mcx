clc;
close all;
clear all;

addpath('../mcx')
addpath('../mcx/mcxlab')


%%
% homo media: maxpop = 1
% 2d image: 100 x 100
% there are 10K diffferent location

% Top-level Dir
topFolderName='../../data/rand2d_homo';
if ~exist('../../data/rand2d_homo/', 'dir')  mkdir(topFolderName); end

% ../../data/rand2d/1e4
dir_phn_noisy = sprintf('%s/%1.0e', topFolderName, 1e4);
if ~exist(dir_phn_noisy, 'dir')  mkdir(dir_phn_noisy); end
   
% ../../data/rand2d/1e8
dir_phn_clean = sprintf('%s/%1.0e', topFolderName, 1e8);
if ~exist(dir_phn_clean, 'dir')  mkdir(dir_phn_clean); end

%%
N = 100 * 100;
% % Generate new random seed for Monte Carlo simulation
% rand_seed = randi([1 2^31-1], 1, N);
% if (length(unique(rand_seed)) < length(rand_seed)) ~= 0
% error('There are repeated random seeds!')
% end

rand_seed = linspace(1, N, N);

srcPos = zeros(N, 2);

pid = 1;
for rowPos = 0: 99
    for colPos = 0 : 99
        srcPos(pid,1) = rowPos;
        srcPos(pid,2) = colPos;
        pid = pid + 1;
    end
end


%%

testID = 1;

for j = 1 : N 
rand_sd = rand_seed(j);

cur_srcPos = [srcPos(testID,1)  srcPos(testID, 2)];

% noisy
[currentImage, ~, ~] = rand_2d_mcx(1e4, 1, [100 100], rand_sd, cur_srcPos);

fname = sprintf('%s/test%d.mat', dir_phn_noisy,  testID);
fprintf('Generating %s\n',fname);
feval('save', fname, 'currentImage');

% clean
[currentImage, ~, ~] = rand_2d_mcx(1e8, 1, [100 100], rand_sd, cur_srcPos);

fname = sprintf('%s/test%d.mat', dir_phn_clean,  testID);
fprintf('Generating %s\n',fname);
feval('save', fname, 'currentImage');

testID = testID + 1;
break
end


