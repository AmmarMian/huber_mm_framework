function [yout,yout_h,yout_v] = hubniht_denoising(y,n,A,c,K,Weight,horizontal)
% Image denoising using HUBNIHT algorithm. 
% 
%  INPUT  
%          y: N x N grayscale image
%          n: (optional) use n x n image patches. Default n = 8;
%          A: (optiona) Dictionary. Default: redundant 2D-DCT dictionary 
%               of 256 atoms. 
%          c: (optional) threshold of Huber's loss. Default c =  0.7317
%               which gives 85 percent efficiency in Gaussian noise  
%          K: sparsity level to be used by HUBNIHT algorithm. Default K=10 
%     Weight: (optional) N x N matrix of weights. If not given, then weights
%            are computed by the function. 
%         y0: (optional) N x N true image of which y is a noisy version of.
%             Naturally y0 is available only in synthetic studies.  When 
%             given, then the function computes the PSNR values as well.             
% horizontal: (logical) true or false. Default = true, in which case image 
%            patches are recovered using horizontal scanning. If false, then
%            both horizontal and vertical scanning are used and
%
% OUTPUT  
%       yout: denoised image. If horizontal = true, then this is output of 
%             the horizontal scan and otherwise average of the horizontal 
%             and vertical scans (yout_h and yout_v)
%     yout_h: denoised image using horizontal scan. Only given (nonempty) 
%               when horizontal = false. 
%     yout_v: denoised image using vertical scan. Only given (nonempty) 
%              when  horizontal = false. 
%      
% Author: Esa Ollila, Aalto University, 2020 
%--------------------------------------------------------------------------

y = double(y);
N = size(y,1);

if nargin < 3 || isempty(A)    
    A = redDCT(n);
end

if nargin < 4 || isempty(c) 
    c = 0.7317; % 85 percent efficiency in gaussian noise
end

if nargin < 5 || isempty(K) 
    K = 10; % 85 percent efficiency in gaussian noise
end

if nargin < 6 || isempty(Weight) 
    Weight = ImageWeights(n,N);
end

if nargin < 7 || isempty(horizontal)
    horizontal = true;
end

%% Extract patches from the noisy image
Imx = N-n+1;
Data1 = zeros(n^2,(N-n+1)^2);
Data2 = zeros(n^2,(N-n+1)^2);
cnt = 1; 
for i = 1:1:Imx
    for j = 1:1:Imx
        patch1 = y(i:i+n-1,j:j+n-1); % horizontal scan data
        patch2 = y(j:j+n-1,i:i+n-1); % vertical scan data 
        Data1(:,cnt) = patch1(:); 
        Data2(:,cnt) = patch2(:); 
        cnt=cnt+1;
    end
end

%% 
supp1in = []; s1in = []; sig1in = []; 
supp2in = []; s2in = []; sig2in = []; 

CoefMat1 = cell(1,Imx);
CoefMat2 = cell(1,Imx);

% Execute the for loop using parallel workers
if horizontal 
    
    parfor i = 1:Imx

        supp1 = supp1in; s1 = s1in; sig1 = sig1in; 
        %supp2 = supp2in; s2 = s2in; sig2 = sig2in; 

        ctmp1 = zeros(size(A,2),Imx);
        %ctmp2 = zeros(size(A,2),Imx);

        D1 = Data1(:,(i-1)*Imx + (1:Imx)); 
        %D2 = Data2(:,(i-1)*Imx + (1:Imx)); 

        for  j = 1:Imx

            %% recover sparse representation
            %-- HUBNIHT row-wise direction

            [s1,sig1,supp1] = hubniht(D1(:,j),A,K,supp1,sig1,s1,c,0);
            ctmp1(:,j) = s1;
            %-- HUBNIHT column-wise direction
            %[s2,sig2,supp2] = hubniht(D2(:,j),A,K,supp2,sig2,s2,c,0);
            %ctmp2(:,j) = s2;

        end  

        CoefMat1{i} = ctmp1;
        %CoefMat2{i}=ctmp2;

    end
    
    CoefMat1 = cell2mat(CoefMat1);
    %CoefMat2 = cell2mat(CoefMat2);

    yout = RecoverImage(N,A,CoefMat1,'horizontal'); 
    %yout_v = RecoverImage(N,A,CoefMat2,'vertical'); 

    yout = yout./Weight; 
    %yout_v = yout_v./Weight; 
    yout_h = []; yout_v = [];
    
else

    parfor i = 1:Imx

        supp1 = supp1in; s1 = s1in; sig1 = sig1in; 
        supp2 = supp2in; s2 = s2in; sig2 = sig2in; 

        ctmp1 = zeros(size(A,2),Imx);
        ctmp2 = zeros(size(A,2),Imx);
        D1 = Data1(:,(i-1)*Imx + (1:Imx)); 
        D2 = Data2(:,(i-1)*Imx + (1:Imx)); 

        for  j = 1:Imx

            %% recover sparse representation
            %-- HUBNIHT row-wise direction
            [s1,sig1,supp1] = hubniht(D1(:,j),A,K,supp1,sig1,s1,c,0);
            ctmp1(:,j) = s1;
            %-- HUBNIHT column-wise direction
            [s2,sig2,supp2] = hubniht(D2(:,j),A,K,supp2,sig2,s2,c,0);
            ctmp2(:,j) = s2;

        end  

        CoefMat1{i}=ctmp1;
        CoefMat2{i}=ctmp2;

    end

    CoefMat1 = cell2mat(CoefMat1);
    CoefMat2 = cell2mat(CoefMat2);

    yout_h = RecoverImage(N,A,CoefMat1,'horizontal'); 
    yout_v = RecoverImage(N,A,CoefMat2,'vertical'); 

    yout_h = yout_h./Weight; 
    yout_v = yout_v./Weight; 

    yout = (yout_h + yout_v)/2;
end


