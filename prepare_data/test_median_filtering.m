clear all;
close all;


caxis = [-3 7];

% load maxV
maxV = 25;

%--------
% test 50
%--------

%noisy input
load('../data/rand2d/1e+05/test50.mat');
img_noisy = currentImage;
figure,imagesc(log10(img_noisy),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5')


filter_img = medfilt2(img_noisy);
figure,imagesc(log10(filter_img),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5-medfilt')


% % applying filter on norm data
% v_log = log(img_noisy + 1);
% v_log_norm = v_log / maxV;
% v_log_norm_filt = medfilt2(v_log_norm);
% v_filt = exp(v_log_norm_filt) - 1;
% figure,imagesc(log10(v_filt),caxis);
% cb = colorbar('northoutside');
% xlabel('mm')
% ylabel('1e5-medfilt-on-norm')


%%

%--------
% test 1
%--------

%noisy input
load('../data/rand2d/1e+05/test1.mat');
img_noisy = currentImage;
figure,imagesc(log10(img_noisy),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5')


filter_img = medfilt2(img_noisy);
figure,imagesc(log10(filter_img),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5-medfilt')


% % applying filter on norm data
% v_log = log(img_noisy + 1);
% v_log_norm = v_log / maxV;
% v_log_norm_filt = medfilt2(v_log_norm);
% v_filt = exp(v_log_norm_filt) - 1;
% figure,imagesc(log10(v_filt),caxis);
% cb = colorbar('northoutside');
% xlabel('mm')
% ylabel('1e5-medfilt-on-norm')