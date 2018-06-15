clear all;
close all;


caxis = [-3 7];

% load maxV

maxV = load('maxV.mat');
maxV = maxV.maxV;

%% test 1
%noisy input
load('../../data/rand2d/1e+05/test1.mat');
img_noisy = currentImage;
figure,imagesc(log10(img_noisy),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5')


% model
load('./test_results/1e5model-test1.mat');

% undo normalization, revert log(x + 1) = y  => x = exp(y) - 1
output_clean = output_clean * maxV;
x = exp(output_clean) - 1;

max(max(x))
min(min(x))

pos = x < 0.0;
x(pos) = 1e-8;

figure,imagesc(log10(x),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5-NN')

% clean 
load('../../data/rand2d/1e+08/test1.mat');

img_clean = currentImage;

figure,imagesc(log10(img_clean),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e8')


%% test 50 
%noisy input
load('../../data/rand2d/1e+05/test50.mat');
img_noisy = currentImage;
figure,imagesc(log10(img_noisy),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5')


% model
load('./test_results/1e5model-test50.mat');

% undo normalization, revert log(x + 1) = y  => x = exp(y) - 1
output_clean = output_clean * maxV;
x = exp(output_clean) - 1;

max(max(x))
min(min(x))

pos = x < 0.0;
x(pos) = 1e-8;

figure,imagesc(log10(x),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e5-NN')

% clean 
load('../../data/rand2d/1e+08/test50.mat');

img_clean = currentImage;

figure,imagesc(log10(img_clean),caxis);
cb = colorbar('northoutside');
xlabel('mm')
ylabel('1e8')

