clear all; close all; clc;

mcxseed=111;

% the test image is grayscle, text is black, bg is white
% size is 100 x 100

%%

% input_img = imread('./images/square.png');  % 
%input_img = imread('./images/square_x15.png');  % 

%img_modify = uint8(input_img < 255); % make the text 1, others bg is 0 255
%img_modify = img_modify + 1; % raise 1 to distinguish from the background

%img_modify = uint8(ones(100,100));
img_modify = ones(100,100);
img_modify(31:70,11:50) = 2; %  40x40, located at 10 along x-axis
img_mofify = uint8(img_modify);

% imagesc(img_modify)

figure;
[cwdata, imgdata, ~] = rand_2d_mcx_grid_test(1e5, img_modify, 123, mcxseed);
subplot(2,2,1);
imagesc(log10(abs(cwdata))); colorbar;
title('1e5');
%currentImage = cwdata; feval('save', 'square_1e5.mat', 'currentImage');


[cwdata1, imgdata1, ~] = rand_2d_mcx_grid_test(1e6, img_modify, 123, mcxseed);
subplot(2,2,2);
imagesc(log10(abs(cwdata1))); colorbar;
title('1e6');
%currentImage = cwdata1; feval('save', 'square_1e6.mat', 'currentImage');

[cwdata2, imgdata2, ~] = rand_2d_mcx_grid_test(1e7, img_modify, 123, mcxseed);
subplot(2,2,3);
imagesc(log10(abs(cwdata2))); colorbar;
title('1e7');
%currentImage = cwdata2; feval('save', 'square_1e7.mat', 'currentImage');


[cwdata3, imgdata3, ~] = rand_2d_mcx_grid_test(1e8, img_modify, 123, mcxseed);
subplot(2,2,4);
imagesc(log10(abs(cwdata3))); colorbar;
title('1e8');
%currentImage = cwdata3; feval('save', 'square_1e8.mat', 'currentImage');

