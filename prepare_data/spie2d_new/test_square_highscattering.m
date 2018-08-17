clear all; close all; clc;

mcxseed=111;

% the test image is grayscle, text is black, bg is white
% size is 100 x 100

%%

input_img = imread('./images/square.png');  % 
img_modify = uint8(input_img < 255); % make the text 1, others bg is 0 255

img_modify = img_modify + 1; % raise 1 to distinguish from the background
% imagesc(img_modify)

figure;
[cwdata, imgdata, ~] = rand_2d_mcx_grid_test(1e5, img_modify, 123, mcxseed);
subplot(2,2,1);
imagesc(log10(abs(cwdata))); colorbar;
currentImage = cwdata; feval('save', 'square_1e5_scat100.mat', 'currentImage');


[cwdata1, imgdata1, ~] = rand_2d_mcx_grid_test(1e6, img_modify, 123, mcxseed);
subplot(2,2,2);
imagesc(log10(abs(cwdata1))); colorbar;
currentImage = cwdata1; feval('save', 'square_1e6_scat100.mat', 'currentImage');

[cwdata2, imgdata2, ~] = rand_2d_mcx_grid_test(1e7, img_modify, 123, mcxseed);
subplot(2,2,3);
imagesc(log10(abs(cwdata2))); colorbar;
currentImage = cwdata2; feval('save', 'square_1e7_scat100.mat', 'currentImage');


[cwdata3, imgdata3, ~] = rand_2d_mcx_grid_test(1e8, img_modify, 123, mcxseed);
subplot(2,2,4);
imagesc(log10(abs(cwdata3))); colorbar;
currentImage = cwdata3; feval('save', 'square_1e8_scat100.mat', 'currentImage');


%%

% input_img = imread('./images/square01.png');  % 
% img_modify = uint8(input_img < 255); % make the text 1, others bg is 0 255
% 
% img_modify = img_modify + 1; % raise 1 to distinguish from the background
% % imagesc(img_modify)
% 
% figure;
% [cwdata, imgdata, ~] = rand_2d_mcx_grid_test(1e5, img_modify, 123, mcxseed);
% subplot(1,3,1);
% imagesc(log10(abs(cwdata))); colorbar;
% currentImage = cwdata; feval('save', 'square01_1e5.mat', 'currentImage');
% 
% 
% [cwdata1, imgdata1, ~] = rand_2d_mcx_grid_test(1e7, img_modify, 123, mcxseed);
% subplot(1,3,2);
% imagesc(log10(abs(cwdata1))); colorbar;
% currentImage = cwdata; feval('save', 'square01_1e7.mat', 'currentImage');
% 
% [cwdata1, imgdata1, ~] = rand_2d_mcx_grid_test(1e8, img_modify, 123, mcxseed);
% subplot(1,3,3);
% imagesc(log10(abs(cwdata1))); colorbar;
% currentImage = cwdata; feval('save', 'square01_1e8.mat', 'currentImage');

%%

% input_img = imread('./images/square02.png');  % 
% img_modify = uint8(input_img < 255); % make the text 1, others bg is 0 255
% 
% img_modify = img_modify + 1; % raise 1 to distinguish from the background
% % imagesc(img_modify)
% 
% figure;
% [cwdata, imgdata, ~] = rand_2d_mcx_grid_test(1e5, img_modify, 123, mcxseed);
% subplot(2,2,1);
% imagesc(log10(abs(cwdata))); colorbar;
% currentImage = cwdata; feval('save', 'square02_1e5.mat', 'currentImage');
% 
% 
% [cwdata1, imgdata1, ~] = rand_2d_mcx_grid_test(1e6, img_modify, 123, mcxseed);
% subplot(2,2,2);
% imagesc(log10(abs(cwdata1))); colorbar;
% currentImage = cwdata1; feval('save', 'square02_1e6.mat', 'currentImage');
% 
% [cwdata2, imgdata2, ~] = rand_2d_mcx_grid_test(1e7, img_modify, 123, mcxseed);
% subplot(2,2,3);
% imagesc(log10(abs(cwdata2))); colorbar;
% currentImage = cwdata2; feval('save', 'square02_1e7.mat', 'currentImage');
% 
% 
% [cwdata3, imgdata3, ~] = rand_2d_mcx_grid_test(1e8, img_modify, 123, mcxseed);
% subplot(2,2,4);
% imagesc(log10(abs(cwdata3))); colorbar;
% currentImage = cwdata3; feval('save', 'square02_1e8.mat', 'currentImage');



%%

% input_img = imread('./images/square03.png');  % 
% img_modify = uint8(input_img < 255); % make the text 1, others bg is 0 255
% 
% img_modify = img_modify + 1; % raise 1 to distinguish from the background
% % imagesc(img_modify)
% 
% figure;
% [cwdata, imgdata, ~] = rand_2d_mcx_grid_test(1e5, img_modify, 123, mcxseed);
% subplot(1,3,1);
% imagesc(log10(abs(cwdata))); colorbar;
% currentImage = cwdata; feval('save', 'square03_1e5.mat', 'currentImage');
% 
% 
% [cwdata1, imgdata1, ~] = rand_2d_mcx_grid_test(1e7, img_modify, 123, mcxseed);
% subplot(1,3,2);
% imagesc(log10(abs(cwdata1))); colorbar;
% currentImage = cwdata; feval('save', 'square03_1e7.mat', 'currentImage');
% 
% [cwdata1, imgdata1, ~] = rand_2d_mcx_grid_test(1e8, img_modify, 123, mcxseed);
% subplot(1,3,3);
% imagesc(log10(abs(cwdata1))); colorbar;
% currentImage = cwdata; feval('save', 'square03_1e8.mat', 'currentImage');
