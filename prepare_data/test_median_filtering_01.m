clear all;
close all;


caxis = [-3 7];

figure;

% noisy
load('../data/rand2d/1e+05/test50.mat');
img_noisy = currentImage;
subplot(2,3,1),imagesc(log10(img_noisy),caxis);
% cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5')

load('../data/rand2d/1e+08/test50.mat');
img_clean = currentImage;
subplot(2,3,2),imagesc(log10(img_clean),caxis);
xlabel('mm')
ylabel('1e8')


img_residual = img_clean - img_noisy;
idx = img_residual <= 0;
img_residual(idx) = 1e-8; % apply flooring before log10()
subplot(2,3,3),imagesc(log10(img_residual),caxis);
xlabel('mm')
ylabel('Residual (clean - noisy)')


filter_residual = medfilt2(img_residual);
subplot(2,3,4),imagesc(log10(filter_residual),caxis);
xlabel('mm')
ylabel('Filtered Residual')


noisy_sub_filterRes = img_noisy - filter_residual;
idx = noisy_sub_filterRes <= 0;
noisy_sub_filterRes(idx) = 1e-8; % apply flooring before log10()
subplot(2,3,5),imagesc(log10(noisy_sub_filterRes),caxis);
xlabel('mm')
ylabel('1e5 - filterResidual')


%%

figure;

% noisy
load('../data/rand2d/1e+05/test1.mat');
img_noisy = currentImage;
subplot(2,3,1),imagesc(log10(img_noisy),caxis);
% cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5')

load('../data/rand2d/1e+08/test1.mat');
img_clean = currentImage;
subplot(2,3,2),imagesc(log10(img_clean),caxis);
xlabel('mm')
ylabel('1e8')


img_residual = img_clean - img_noisy;
idx = img_residual <= 0;
img_residual(idx) = 1e-8; % apply flooring before log10()
subplot(2,3,3),imagesc(log10(img_residual),caxis);
xlabel('mm')
ylabel('Residual (clean - noisy)')


filter_residual = medfilt2(img_residual);
subplot(2,3,4),imagesc(log10(filter_residual),caxis);
xlabel('mm')
ylabel('Filtered Residual')


noisy_sub_filterRes = img_noisy - filter_residual;
idx = noisy_sub_filterRes <= 0;
noisy_sub_filterRes(idx) = 1e-8; % apply flooring before log10()
subplot(2,3,5),imagesc(log10(noisy_sub_filterRes),caxis);
xlabel('mm')
ylabel('1e5 - filterResidual')



%%

figure;

% noisy
load('../data/rand2d/1e+05/test900.mat');
img_noisy = currentImage;
subplot(2,3,1),imagesc(log10(img_noisy),caxis);
% cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5')

load('../data/rand2d/1e+08/test900.mat');
img_clean = currentImage;
subplot(2,3,2),imagesc(log10(img_clean),caxis);
xlabel('mm')
ylabel('1e8')


img_residual = img_clean - img_noisy;
idx = img_residual <= 0;
img_residual(idx) = 1e-8; % apply flooring before log10()
subplot(2,3,3),imagesc(log10(img_residual),caxis);
xlabel('mm')
ylabel('Residual (clean - noisy)')


filter_residual = medfilt2(img_residual);
subplot(2,3,4),imagesc(log10(filter_residual),caxis);
xlabel('mm')
ylabel('Filtered Residual')


noisy_sub_filterRes = img_noisy - filter_residual;
idx = noisy_sub_filterRes <= 0;
noisy_sub_filterRes(idx) = 1e-8; % apply flooring before log10()
subplot(2,3,5),imagesc(log10(noisy_sub_filterRes),caxis);
xlabel('mm')
ylabel('1e5 - filterResidual')


%%

figure;

% noisy
load('../data/osa/1e+05/1/y/osa_phn1e+05_test1_img50.mat');
img_noisy = currentImage;
subplot(2,3,1),imagesc(log10(img_noisy),caxis);
% cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5')

load('../data/osa/1e+08/1/y/osa_phn1e+08_test1_img50.mat');
img_clean = currentImage;
subplot(2,3,2),imagesc(log10(img_clean),caxis);
xlabel('mm')
ylabel('1e8')


img_residual = img_clean - img_noisy;
idx = img_residual <= 0;
img_residual(idx) = 1e-8; % apply flooring before log10()
subplot(2,3,3),imagesc(log10(img_residual),caxis);
xlabel('mm')
ylabel('Residual (clean - noisy)')


filter_residual = medfilt2(img_residual);
subplot(2,3,4),imagesc(log10(filter_residual),caxis);
xlabel('mm')
ylabel('Filtered Residual')


noisy_sub_filterRes = img_noisy - filter_residual;
idx = noisy_sub_filterRes <= 0;
noisy_sub_filterRes(idx) = 1e-8; % apply flooring before log10()
subplot(2,3,5),imagesc(log10(noisy_sub_filterRes),caxis);
xlabel('mm')
ylabel('1e5 - filterResidual')


%%

figure;

% noisy
load('../data/osa/1e+05/1/y/osa_phn1e+05_test1_img1.mat');
img_noisy = currentImage;
subplot(2,3,1),imagesc(log10(img_noisy),caxis);
% cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5')

load('../data/osa/1e+08/1/y/osa_phn1e+08_test1_img1.mat');
img_clean = currentImage;
subplot(2,3,2),imagesc(log10(img_clean),caxis);
xlabel('mm')
ylabel('1e8')


img_residual = img_clean - img_noisy;
idx = img_residual <= 0;
img_residual(idx) = 1e-8; % apply flooring before log10()
subplot(2,3,3),imagesc(log10(img_residual),caxis);
xlabel('mm')
ylabel('Residual (clean - noisy)')


filter_residual = medfilt2(img_residual);
subplot(2,3,4),imagesc(log10(filter_residual),caxis);
xlabel('mm')
ylabel('Filtered Residual')


noisy_sub_filterRes = img_noisy - filter_residual;
idx = noisy_sub_filterRes <= 0;
noisy_sub_filterRes(idx) = 1e-8; % apply flooring before log10()
subplot(2,3,5),imagesc(log10(noisy_sub_filterRes),caxis);
xlabel('mm')
ylabel('1e5 - filterResidual')


