clear all; close all; clc;

mcxseed=111;

% the test image is grayscle, text is black, bg is transparent
% size is 100 x 100
input_img = imread('./images/test_c.png');  
img_modify = uint8(input_img == 0); % make the text 1, others 0
img_modify = img_modify + 1; % raise 1 to distinguish from the background
% imagesc(img_modify)

figure;
[cwdata, imgdata, ~] = rand_2d_mcx_grid_test(1e4, img_modify, 123, mcxseed);
subplot(1,2,1);
imagesc(log10(abs(cwdata))); colorbar;

[cwdata1, imgdata1, ~] = rand_2d_mcx_grid_test(1e7, img_modify, 123, mcxseed);
subplot(1,2,2);
imagesc(log10(abs(cwdata1))); colorbar;


%%

% the test image is grayscle, text is black, bg is transparent
% size is 100 x 100
input_img = imread('./images/test_o.png');  
img_modify = uint8(input_img == 0); % make the text 1, others 0
img_modify = img_modify + 1; % raise 1 to distinguish from the background
% imagesc(img_modify)

figure;
[cwdata, imgdata, ~] = rand_2d_mcx_grid_test(1e4, img_modify, 123, mcxseed);
subplot(1,2,1);
imagesc(log10(abs(cwdata))); colorbar;

[cwdata1, imgdata1, ~] = rand_2d_mcx_grid_test(1e7, img_modify, 123, mcxseed);
subplot(1,2,2);
imagesc(log10(abs(cwdata1))); colorbar;