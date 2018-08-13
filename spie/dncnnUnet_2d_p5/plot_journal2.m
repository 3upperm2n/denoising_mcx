clear all;
close all;


caxis = [-3 7];

% load maxV

maxV = load('maxV.mat');
maxV = maxV.maxV;

%% jounal2  hom   square

figure

%noisy input
img_noisy = load('../../prepare_data/spie2d_customize/journal2_hom/square_1e5.mat');
img_noisy = img_noisy.currentImage;

subplot(1,3,1); imagesc(log10(img_noisy),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5')


% model
load('./test_results/j2-hom-square-p5.mat');

% undo normalization, revert log(x + 1) = y  => x = exp(y) - 1
output_clean = squeeze(output_clean) * maxV;
x = exp(output_clean) - 1;

max(max(x))
min(min(x))

pos = x < 0.0;
x(pos) = 1e-8;

subplot(1,3,2);imagesc(log10(x),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5-NN')


% clean 

img_clean = load('../../prepare_data/spie2d_customize/journal2_hom/square_1e7.mat');
img_clean = img_clean.currentImage;
%img_clean = currentImage;

subplot(1,3,3); imagesc(log10(img_clean),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e7')

% Improved SNR
tmp = abs(img_noisy - img_clean);
sum_noisy = sum(sum(tmp));

tmp1 = abs(x - img_clean);
sum_nn = sum(sum(tmp1));

isnr = (log10(sum_noisy) - log10(sum_nn)) * 20;

fprintf('The ISNR = %.3f.\n',isnr);



%% jounal2  hom   square02

figure

%noisy input
img_noisy = load('../../prepare_data/spie2d_customize/journal2_hom/square02_1e5.mat');
img_noisy = img_noisy.currentImage;

subplot(1,3,1); imagesc(log10(img_noisy),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5')


% model
load('./test_results/j2-hom-square02-p5.mat');

% undo normalization, revert log(x + 1) = y  => x = exp(y) - 1
output_clean = squeeze(output_clean) * maxV;
x = exp(output_clean) - 1;

max(max(x))
min(min(x))

pos = x < 0.0;
x(pos) = 1e-8;

subplot(1,3,2);imagesc(log10(x),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5-NN')


% clean 

img_clean = load('../../prepare_data/spie2d_customize/journal2_hom/square02_1e7.mat');
img_clean = img_clean.currentImage;
%img_clean = currentImage;

subplot(1,3,3); imagesc(log10(img_clean),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e7')

% Improved SNR
tmp = abs(img_noisy - img_clean);
sum_noisy = sum(sum(tmp));

tmp1 = abs(x - img_clean);
sum_nn = sum(sum(tmp1));

isnr = (log10(sum_noisy) - log10(sum_nn)) * 20;

fprintf('The ISNR = %.3f.\n',isnr);


%% absorber square 


figure

%noisy input
img_noisy = load('../../prepare_data/spie2d_customize/journal2_absorber/square_1e5.mat');
img_noisy = img_noisy.currentImage;

subplot(1,3,1); imagesc(log10(img_noisy),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5')


% model
load('./test_results/j2-absorber-square-p5.mat');

% undo normalization, revert log(x + 1) = y  => x = exp(y) - 1
output_clean = squeeze(output_clean) * maxV;
x = exp(output_clean) - 1;

max(max(x))
min(min(x))

pos = x < 0.0;
x(pos) = 1e-8;

subplot(1,3,2);imagesc(log10(x),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5-NN')


% clean 

img_clean = load('../../prepare_data/spie2d_customize/journal2_absorber/square_1e7.mat');
img_clean = img_clean.currentImage;
%img_clean = currentImage;

subplot(1,3,3); imagesc(log10(img_clean),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e7')

% Improved SNR
tmp = abs(img_noisy - img_clean);
sum_noisy = sum(sum(tmp));

tmp1 = abs(x - img_clean);
sum_nn = sum(sum(tmp1));

isnr = (log10(sum_noisy) - log10(sum_nn)) * 20;

fprintf('The ISNR = %.3f.\n',isnr);




%% absorber square02


figure

%noisy input
img_noisy = load('../../prepare_data/spie2d_customize/journal2_absorber/square02_1e5.mat');
img_noisy = img_noisy.currentImage;

subplot(1,3,1); imagesc(log10(img_noisy),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5')


% model
load('./test_results/j2-absorber-square02-p5.mat');

% undo normalization, revert log(x + 1) = y  => x = exp(y) - 1
output_clean = squeeze(output_clean) * maxV;
x = exp(output_clean) - 1;

max(max(x))
min(min(x))

pos = x < 0.0;
x(pos) = 1e-8;

subplot(1,3,2);imagesc(log10(x),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5-NN')


% clean 

img_clean = load('../../prepare_data/spie2d_customize/journal2_absorber/square02_1e7.mat');
img_clean = img_clean.currentImage;
%img_clean = currentImage;

subplot(1,3,3); imagesc(log10(img_clean),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e7')

% Improved SNR
tmp = abs(img_noisy - img_clean);
sum_noisy = sum(sum(tmp));

tmp1 = abs(x - img_clean);
sum_nn = sum(sum(tmp1));

isnr = (log10(sum_noisy) - log10(sum_nn)) * 20;

fprintf('The ISNR = %.3f.\n',isnr);

