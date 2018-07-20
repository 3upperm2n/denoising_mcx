clc;
close all;
clear all;

addpath('../mcx')
addpath('../mcx/mcxlab')


% homo media: maxpop = 0
% 2d image: 100 x 100

% Top-level Dir
topFolderName='../../data/spie2d/hom';
if ~exist('../../data/spie2d/hom', 'dir')  mkdir(topFolderName); end

% ../../data/spie2d/hom/1e4
dir_phn_1e4 = sprintf('%s/%1.0e', topFolderName, 1e4);
if ~exist(dir_phn_1e4, 'dir')  mkdir(dir_phn_1e4); end
   
% ../../data/spie2d/hom/1e5
dir_phn_1e5 = sprintf('%s/%1.0e', topFolderName, 1e5);
if ~exist(dir_phn_1e5, 'dir')  mkdir(dir_phn_1e5); end

% ../../data/spie2d/hom/1e6
dir_phn_1e6 = sprintf('%s/%1.0e', topFolderName, 1e6);
if ~exist(dir_phn_1e6, 'dir')  mkdir(dir_phn_1e6); end

% ../../data/spie2d/hom/1e7
dir_phn_1e7 = sprintf('%s/%1.0e', topFolderName, 1e7);
if ~exist(dir_phn_1e7, 'dir')  mkdir(dir_phn_1e7); end

% ../../data/spie2d/hom/1e8
dir_phn_1e8 = sprintf('%s/%1.0e', topFolderName, 1e8);
if ~exist(dir_phn_1e8, 'dir')  mkdir(dir_phn_1e8); end

%
% we will generate 1000 x 20 = 20K images
% 

N = 1000 * 20;

% Generate new random seed for Monte Carlo simulation
rand_seed = randi([1 2^31-1], 1, N);
if (length(unique(rand_seed)) < length(rand_seed)) ~= 0
error('There are repeated random seeds!')
end


testID = 1;
% 20 different set for opt objs
for i = 1 : N
    rand_sd = rand_seed(i);

    % 1e4 
    [currentImage, ~, ~] = rand_2d_mcx(1e4, 0, [100 100], rand_sd);
    fname = sprintf('%s/test%d.mat', dir_phn_1e4,  testID);
    fprintf('Generating %s\n',fname);
    feval('save', fname, 'currentImage');
    
    % 1e5 
    [currentImage, ~, ~] = rand_2d_mcx(1e5, 0, [100 100], rand_sd);
    fname = sprintf('%s/test%d.mat', dir_phn_1e5,  testID);
    fprintf('Generating %s\n',fname);
    feval('save', fname, 'currentImage');
    
    % 1e6
    [currentImage, ~, ~] = rand_2d_mcx(1e6, 0, [100 100], rand_sd);
    fname = sprintf('%s/test%d.mat', dir_phn_1e6,  testID);
    fprintf('Generating %s\n',fname);
    feval('save', fname, 'currentImage');

    % 1e7
    [currentImage, ~, ~] = rand_2d_mcx(1e7, 0, [100 100], rand_sd);
    fname = sprintf('%s/test%d.mat', dir_phn_1e7,  testID);
    fprintf('Generating %s\n',fname);
    feval('save', fname, 'currentImage');

    % 1e8
    [currentImage, ~, ~] = rand_2d_mcx(1e8, 0, [100 100], rand_sd);
    fname = sprintf('%s/test%d.mat', dir_phn_1e8,  testID);
    fprintf('Generating %s\n',fname);
    feval('save', fname, 'currentImage');

    testID = testID + 1;

    %break
end