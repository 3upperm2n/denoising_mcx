% simple test

clc;
close all;
clear all;

figure;
subplot(1,2,1);
load('../../data/spie2d/het/1e+04/test1.mat');
p4img = currentImage;
imagesc(log10(abs(p4img)));

subplot(1,2,2);
load('../../data/spie2d/het/1e+07/test1.mat');
p7img = currentImage;
imagesc(log10(abs(p7img)));