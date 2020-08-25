close all;
clc;
clearvars;
% is you are in the examples folder, then add these to the paths
addpath('../');
addpath('../pics');
addpath('../aux');

%% Possible choices for Huber's threshold 
c95 = 1.345;
c85 = 0.7317;
c80 = 0.5294;
c75 = 0.3528783;
c70 = 0.1917;

%% Create noisy image 
load lena256color.mat;
y0 = double(rgb2gray(X0));
clear X0;
y0 = y0/max(y0(:));  % making the pixel values lie in the interval [0,1]
n = 8; % use 8 by 8 windows

%%  Create noisy image 
rng default;
y = imnoise(y0,'salt & pepper',0.1); % ADD SALT AND PEPPER NOISE
maxI = max(y(:));
N = size(y,1); 

%% Plot the true and noisy images

h = figure(1); clf;
imagesc(y0); 
colormap(gray(256)); 
axis image; 
axis off; 
title('Original Image');

h=figure(2); clf;
imagesc(y); 
colormap(gray(256)); 
axis image; 
axis off; 

PSNRin = psnr(y,y0,maxI);
title(['Noisy Image with PSNR = ',num2str(PSNRin),' dB'],'FontSize',16);

%% Median filter denoising 
youtMF = medfilt2(y,[3 3]);
PSNR_MF = psnr(youtMF,y0,maxI)

h=figure(3); clf;
imagesc(youtMF); 
colormap(gray(256)); 
axis image; 
axis off; 
title(['Median Filter PSNR = ',num2str(PSNR_MF),' dB'],'FontSize',16);

%% HUBNIHT 
K = 11; % sparsity level to consider
horizontal = false; % denoise by horizontal scan of patches 
c = c70; % threshold for Huber's loss function

% Note: hubniht_denoising function uses parallel pool to speed up the computation
% horizontal = false implies that scanning in patches is performed  both in 
% horizontal and vertical directions. Faster way is to put horizontal = true, 
% in which case vertical scan is not done --> this makes computing twice faster
% On my MacBook Pro laptop it took 200second to run the code below. 
tic;
[yout,yout_h,yout_v] = hubniht_denoising(y,n,[],c,K,[],horizontal) ;
toc;
PSNR(1) = psnr(yout,y0,maxI); 
if ~horizontal
    PSNR(2) = psnr(yout_h,y0,maxI); % horizontal 
    PSNR(3) = psnr(yout_v,y0,maxI);
end

%% Plot the denoised image with the best PSNR
figure(4); clf;
imagesc(yout); 
colormap(gray(256)); 
axis image; 
axis off; 
title(['HUBNIHT PSNR = ',num2str(PSNR(1)),' dB'],'FontSize',16);

