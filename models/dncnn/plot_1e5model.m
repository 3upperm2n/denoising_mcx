clear all;
close all;


caxis = [-3 7];


%%  1e5

% %------------
% % image 1
% %------------
% %---
% %noisy input
% %---
% load('../data/osa_data/1e+05/1/osa_phn1e+05_test1_img1.mat');
% img_noisy = currentImage;
% figure,imagesc(log10(img_noisy),caxis);
% 
% %colorbar
% %xlabel('z axis'),ylabel('x axis')
% %title('image 1 (10^5) : noisy')     % check the image id
% cb = colorbar('northoutside');
% xlabel('mm')
% 
% 
% %---
% % model input
% %---
% 
% load('./test_results/1e5model-1e5-log_test1_img1.mat');
% 
% % revert log(x + 1) = y  => x = exp(y) - 1
% x = exp(output_clean) - 1;
% 
% max(max(x))
% min(min(x))
% 
% pos = x < 0.0;
% x(pos) = 1e-8;
% 
% figure,imagesc(log10(x),caxis);
% %colorbar
% %xlabel('z axis'),ylabel('x axis')
% %title('image 1 (10^5) with log : 1e5model ')
% 
% cb = colorbar('northoutside');
% xlabel('mm')
% 
% 
% %---
% % clean 
% %---
% load('../data/osa_data/1e+09/osa_1e9_img1.mat');
% img_clean = currentImage;
% 
% figure,imagesc(log10(img_clean),caxis);
% %colorbar
% %xlabel('z axis'),ylabel('x axis')
% %title('image 1 (10^9) : clean')
% 
% cb = colorbar('northoutside');
% xlabel('mm')


%------------
% image 50 
%------------

%---
%noisy input
%---
load('../../data/osa/1e+05/1/y/osa_phn1e+05_test1_img50.mat');
img_noisy = currentImage;
figure,imagesc(log10(img_noisy),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5')


%---
% model
%---

load('./test_results/1e5model-test1_img50.mat');

% revert log(x + 1) = y  => x = exp(y) - 1
x = exp(output_clean) - 1;

max(max(x))
min(min(x))

pos = x < 0.0;
x(pos) = 1e-8;

figure,imagesc(log10(x),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5-NN')

%---
% clean 
%---
% load('../../data/osa/1e+09/2/y/osa_phn1e+09_test2_img50.mat');
load('../../data/osa/1e+08/1/y/osa_phn1e+08_test1_img50.mat');

img_clean = currentImage;

figure,imagesc(log10(img_clean),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e8')


%
% image 0
%

%---
%noisy input
%---
load('../../data/osa/1e+05/1/y/osa_phn1e+05_test1_img1.mat');
img_noisy = currentImage;
figure,imagesc(log10(img_noisy),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5')


%---
% model input
%---

load('./test_results/1e5model-test1_img1.mat');


% revert log(x + 1) = y  => x = exp(y) - 1
x = exp(output_clean) - 1;

max(max(x))
min(min(x))

pos = x < 0.0;
x(pos) = 1e-8;

figure,imagesc(log10(x),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5-NN')

%---
% clean 
%---
load('../../data/osa/1e+09/1/y/osa_phn1e+09_test1_img1.mat');

img_clean = currentImage;

figure,imagesc(log10(img_clean),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e9')



