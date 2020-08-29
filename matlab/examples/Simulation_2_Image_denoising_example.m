%% Image denoising using |HUBNIHT| 
% It is assumed here that you have downloaded the toolbox 
% <https://github.com/AmmarMian/huber_mm_framework> and are now in
% the _./matlab_ folder

%%
% Set-up the path: it is assumed that you are in the _./examples_ folder
close all;
clc;
clearvars;
addpath('../');
addpath('../pics');
addpath('../aux');

%% Possible choices for Huber's threshold. 
% Due to very spiky salt and pepper noise, it is recommended to use a lower
% value for threshold for Huber's loss. A small value indicates increased robustness.
% In this example, we use |c70 = 0.1917|. 
c95 = 1.345;
c85 = 0.7317;
c80 = 0.5294;
c75 = 0.3528783;
c70 = 0.1917;

%% load the |Lena| image and add noise 
load lena256color.mat;
y0 = double(rgb2gray(X0));
clear X0;
y0 = y0/max(y0(:));  % making the pixel values lie in the interval [0,1]
n = 8; % use 8 by 8 windows

rng default;
y = imnoise(y0,'salt & pepper',0.1); % ADD SALT AND PEPPER NOISE
maxI = max(y(:));
N = size(y,1); 

%% Plot the true and noisy images

figure(1); clf;
imagesc(y0); 
colormap(gray(256)); 
axis image; 
axis off; 
title('Original Image');

figure(2); clf;
imagesc(y); 
colormap(gray(256)); 
axis image; 
axis off; 

PSNRin = psnr(y,y0,maxI);
title(['Noisy Image with PSNR = ',num2str(PSNRin),' dB'],'FontSize',16);

%% Median filter denoising 
youtMF = medfilt2(y,[3 3]);
tic;
PSNR_MF = psnr(youtMF,y0,maxI)
toc;
figure(3); clf;
imagesc(youtMF); 
colormap(gray(256)); 
axis image; 
axis off; 
title(['Median Filter PSNR = ',num2str(PSNR_MF),' dB'],'FontSize',16);

%% HUBNIHT denoising
% Here we use _K=11_ as the value for sparsity level. For the chosen
% threshold value _c_, it gaved the best results. Note that |hubniht_denoising| 
% function uses parallel pool to speed up the computation. 


K = 11; % sparsity level to consider
c = c70; % threshold for Huber's loss function

horizontal = false; % denoise by both horizontal and vertical scan of patches 
%%
% Setting up |horizontal = false| implies that scanning of patches is performed  both in 
% horizontal and vertical directions. The output is then average of the scans. For large images, it is recommended to set |horizontal = true|. This 
% speeds up computations roughly by 1/2 as the vertical scan is not performed.  This is 
% illustrated at the end of this script. 
 
tic;
[yout,yout_h,yout_v] = hubniht_denoising(y,n,[],c,K,[],horizontal) ;
toc;
PSNR(1) = psnr(yout,y0,maxI); 
PSNR(2) = psnr(yout_h,y0,maxI); % horizontal 
PSNR(3) = psnr(yout_v,y0,maxI);
PSNR 

%% 
% As can be noted, on my MacBook Pro laptop it took about 80 seconds to 
% obtain the denoised image. 


%% Plot the denoised image obtained by |HUBNIHT|
figure(4); clf;
imagesc(yout); 
colormap(gray(256)); 
axis image; 
axis off; 
title(['HUBNIHT PSNR = ',num2str(PSNR(1)),' dB'],'FontSize',16);

%% HUBNIHT denoising: horizontal scan only 
% Note that the computation time using horizontal scan only is roughly 1/2 
% of the previous run.

horizontal = true; % denoise by horizontal scan of patches 
tic;
[yout,yout_h,yout_v] = hubniht_denoising(y,n,[],c,K,[],horizontal) ;
toc;
psnr(yout,y0,maxI) % same as PSNR(2) computed earlier 
 
%% Reference 
%
% Esa Ollila and Ammar Mian, "Block-wise minimization-majorization
% algorithm for Huber's criterion: sparse learning and applications", in _Proc. IEEE International Workshop on Machine Learning for Signal Processing_ (MLSP, Sep 21-24, 2020, Espoo, Finland.
