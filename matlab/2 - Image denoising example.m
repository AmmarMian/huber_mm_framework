close all;clc;
clearvars;
%%
load lena256color.mat;
X0 = rgb2gray(X0);
maxX0=double(max(max(max(abs(X0)))));
X=double(X0(:,:,:))/maxX0;  % making the pixel values lie in the interval [0,1]

blocksize=8; %size of the patches e.g. 8X8
step=1; %step size, step=blocksize means no overlap, step=1 means maximum overlap
A = wmpdictionary(blocksize^2,'lstcpt',{{'sym4',5},{'coif4',5},'dct'}); %constructing the overcomplete dictionary
A=full(A);
m=size(A,1);
n=size(A,2);
q=size(X,3);

PSNRdb=12; % PSNR (dB) of noisy image
%% THIS IS FOR GAUSSIAN NOISE 
%sigma_v=10^(-PSNRdb/20);
%V=sigma_v*randn(size(X));
%Y=X+V; %add noise in the image

%% THIS IS FO SALT AND PEPPER 
Y = imnoise(X,'salt & pepper',0.1);

trim1=blocksize+1:size(X,1)-blocksize;
trim2=blocksize+1:size(X,2)-blocksize;

figure(1);clf;imshow(X(trim1,trim2,:));
% title('Original Image');

figure(2);clf;imshow(Y(trim1,trim2,:));
Err=Y-X;
PSNR0=-10*log10( mean(Err(:).^2) );
% title(['Noisy Image with PSNR = ',num2str(PSNR0),' dB']);

%% 
k=2; %sparsity
%flavor='residue'; %flavor='residue' means SOMP/RandSOMP stop when norm of the residue becomes small enough
flavor='sparsity';%flavor='sparsity' means SOMP/RandSOMP stop after k iterations

X_hat1=zeros(size(X));
X_hat2=zeros(size(X));
tic;
for i=1:step:size(Y,1)-blocksize+1,
    
    for j=1:step:size(Y,2)-blocksize+1,
        
        %%%%%%%%%%%%%%%%%%% extract image patches %%%%%%%%%%%%%%%%%%%%
        y=Y(i:i+blocksize-1,j:j+blocksize-1,:);
        y=reshape(y,[blocksize^2 q]);
        
        %%%%%%%%%%%%%% %recover sparse representation using SOMP %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %if (strcmp(flavor,'sparsity'))
        %    s1 = SOMP2(A,y,'sparsity',k);
        %else
        %    s1 = SOMP2(A,y,'residue',c*sqrt(numel(y))*sigma_v);
        %end
        s1 = sniht(y,A,k);
        s2 = hubniht(y,A,k,[],1.345,0);

        %%%%%%%%%%%%%%%%%%%%%%%% construct noise free patches from their sparse representations %%%%%%%%%%%%
        x_hat1=reshape(A*s1,[blocksize blocksize q]);
        x_hat2=reshape(A*s2,[blocksize blocksize q]);
        
        %%%%%%%%%%%%%%%%%%%%%%%% construct the whole noise free image from its patches %%%%%%%%%%%%
        X_hat1(i:i+blocksize-1,j:j+blocksize-1,:)=X_hat1(i:i+blocksize-1,j:j+blocksize-1,:)+x_hat1;
        X_hat2(i:i+blocksize-1,j:j+blocksize-1,:)=X_hat2(i:i+blocksize-1,j:j+blocksize-1,:)+x_hat2;

    end
    if mod(i,5)==0, fprintf('%d ',i); end 
end
beep; beep; beep;
toc;

X_hat1=X_hat1*(step/blocksize)^2;
X_hat2=X_hat2*(step/blocksize)^2;

figure;
imshow(X_hat1(trim1,trim2,:));
Err1=X_hat1(trim1,trim2)-X(trim1,trim2);
PSNR1=-10*log10(mean(Err1(:).^2))
title(['SOMP output with k = ', num2str(k), ' PSNR = ',num2str(PSNR1),' dB']);


figure;
imshow(X_hat2(trim1,trim2,:));
Err2=X_hat2(trim1,trim2)-X(trim1,trim2);
PSNR2=-10*log10(mean(Err2(:).^2));
title(['HUBSNIHT output with k = ', num2str(k), ' PSNR = ',num2str(PSNR2),' dB']);


Diff=X_hat2(trim1,trim2,:)-X(trim1,trim2,:);
Diff=Diff+max(max(max(-Diff)));
Diff=Diff/max(max(max(Diff)));
figure;imshow(Diff);
title('Difference between HUB-SNIHT and original');


Diff=X_hat1(trim1,trim2,:)-X(trim1,trim2,:);
Diff=Diff+max(max(max(-Diff)));
Diff=Diff/max(max(max(Diff)));
figure;imshow(Diff);
title('Difference between SOMP and original');






