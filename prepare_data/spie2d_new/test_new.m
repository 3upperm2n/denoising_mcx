clear all; close all; clc;

mcxseed=947523947;

% the test image is grayscle, text is black, bg is white
% size is 100 x 100

%%

% input_img = imread('./images/square.png');  % 
%input_img = imread('./images/square_x15.png');  % 

%img_modify = uint8(input_img < 255); % make the text 1, others bg is 0 255
%img_modify = img_modify + 1; % raise 1 to distinguish from the background

% img_modify = uint8(ones(100,100));

% imagesc(img_modify)

% figure;
% [cwdata, imgdata, ~] = rand_2d_mcx_grid_test(1e5, img_modify, 123, mcxseed);
% subplot(2,2,1);
% imagesc(log10(abs(cwdata))); colorbar;
% currentImage = cwdata; feval('save', 'square_1e5.mat', 'currentImage');


%%



%% rand square
for ii = 1 :1
img = ones(100,100);
% c_r= randi(80) + 10;
% c_l= randi(80) + 10;
% h = 5+randi(5);
% img(c_r-h:c_r+h,c_l-h:c_l+h)=2; % rand square
% img = uint8(img);
 img(31:70,11:50) = 2; %  40x40, located at 10 along x-axis

%figure,imagesc(img)

[cwdata] = rand_2d_mcx_grid_test(1e7, img, randi(5555), randi(mcxseed));
figure; 
subplot(1,2,1)
imagesc(img)
subplot(1,2,2)
imagesc(log10(abs(cwdata))); %colorbar;
end
